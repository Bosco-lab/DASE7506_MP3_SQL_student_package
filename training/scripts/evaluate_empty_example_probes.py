"""Evaluate Step 3a (relationship/null probes) and Step 3b (empty-example vote alignment) on 4,017 rival pairs and 152 empty tasks."""
from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
COND = ROOT / 'delivery/goal201/conditions04'
COND_SUB = COND / 'submission'
COND_LIB = COND_SUB / 'lib'
sys.path[:0] = [str(ROOT), str(COND_SUB), str(COND_LIB)]

from mp3_sql.common import load_bank, result_fingerprint, write_json
from mp3_sql.score import load_cases
from mp3_sql.sqlite_tool import execute_sql
from parent_agent import _parse_sqlite, header_fidelity, legacy, score_passing_candidate, vote_key


def parse_one(sql: str):
    try:
        ts = sqlglot.parse(sql, read='sqlite')
        return ts[0] if len(ts) == 1 else None
    except Exception:
        return None


def base_tables(tree: exp.Expression, schema_names: dict[str, str]) -> list[str]:
    out = []
    for t in tree.find_all(exp.Table):
        if t.name and t.name.lower() in schema_names and t.name.lower() not in out:
            out.append(t.name.lower())
    return out


def exec_ro(db_path: Path, sql: str):
    try:
        with sqlite3.connect(db_path.as_uri() + '?mode=ro&immutable=1', uri=True) as conn:
            return conn.execute(sql).fetchall()
    except Exception:
        return None


def split_and(node: exp.Expression):
    if isinstance(node, exp.And):
        yield from split_and(node.this)
        yield from split_and(node.expression)
    else:
        yield node


def join_and(terms: list[exp.Expression]):
    cur = terms[0]
    for nxt in terms[1:]:
        cur = exp.And(this=cur, expression=nxt)
    return cur


def strip_all_where(tree: exp.Expression) -> str | None:
    if not isinstance(tree, exp.Select):
        return None
    n = tree.copy()
    n.set('limit', None)
    n.set('offset', None)
    n.set('where', None)
    n.set('having', None)
    return n.sql(dialect='sqlite')


def strip_literal_atoms(tree: exp.Expression) -> str | None:
    if not isinstance(tree, exp.Select):
        return None
    n = tree.copy()
    n.set('limit', None)
    n.set('offset', None)
    for clause in ('where', 'having'):
        c = n.args.get(clause)
        if not c:
            continue
        kept = [term.copy() for term in split_and(c.this) if not term.find(exp.Literal)]
        if kept:
            n.set(clause, exp.Where(this=join_and(kept)) if clause == 'where' else exp.Having(this=join_and(kept)))
        else:
            n.set(clause, None)
    return n.sql(dialect='sqlite')


def has_irreflexive_self_cmp(tree: exp.Expression) -> bool:
    if not isinstance(tree, exp.Select):
        return False
    aliases = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    for node in tree.find_all(exp.GT, exp.LT, exp.NEQ):
        if isinstance(node.this, exp.Column) and isinstance(node.expression, exp.Column):
            lt = aliases.get(node.this.table.lower(), node.this.table.lower())
            rt = aliases.get(node.expression.table.lower(), node.expression.table.lower())
            if node.this.name.lower() == node.expression.name.lower() and lt == rt:
                return True
    return False


def aligned_vote_key(sql: str, schema_info: dict) -> str:
    tree = _parse_sqlite(sql)
    if tree is None:
        return vote_key(sql)
    tree = tree.copy()
    # Canonicalize COUNT(pk) -> COUNT(*)
    pks = {
        (tname, pk.lower())
        for tname, tinfo in schema_info.get('tables', {}).items()
        for pk in tinfo.get('pks', [])
    }
    aliases = {t.alias_or_name.lower(): t.name.lower() for t in tree.find_all(exp.Table) if t.name}
    for cnt in tree.find_all(exp.Count):
        if isinstance(cnt.this, exp.Column):
            tname = aliases.get(cnt.this.table.lower(), '')
            if not tname and len(aliases) == 1:
                tname = next(iter(aliases.values()))
            if (tname, cnt.this.name.lower()) in pks:
                cnt.set('this', exp.Star())
    # Canonicalize single-table IN (SELECT col FROM T ...) when possible
    return vote_key(tree.sql(dialect='sqlite'))


def main():
    started = time.perf_counter()
    pairs_doc = json.loads((ROOT / 'outputs/plan4/empty_rival_pairs.json').read_bytes())
    banks = {}
    cases_map = {}
    replays = {}
    for split in ('train', 'validation'):
        rows, manifest, folder = load_bank(split)
        cases, case_root = load_cases(split, folder)
        banks[split] = ({r['id']: r for r in rows}, manifest, folder)
        cases_map[split] = (cases, case_root)
        rdir = COND / 'evidence' / f'{split}-replay'
        preds = json.loads((rdir / f'{split}.json').read_bytes())['predictions']
        scores = json.loads((rdir / f'{split}-score.json').read_bytes())['per_question']
        replays[split] = {p['id']: (p, g) for p, g in zip(preds, scores, strict=True)}

    cache = {}
    exec_cache = {}

    def grade_3inst(row, split, sql):
        _, manifest, _ = banks[split]
        cases, case_root = cases_map[split]
        tables = [t['name'] for t in manifest['databases'][row['db_id']]['schema']]
        for case in cases['tasks'][row['id']]:
            path = case_root / case['path']
            key = (case['database_sha256'], sql, False)
            res = exec_cache.get(key)
            if res is None:
                res = execute_sql(path, sql, tables, False)
                exec_cache[key] = res
            if not (res['ok'] and result_fingerprint(res, row['ordered']) == case['expected']):
                return False
        return True

    def check_3a(db_path: Path, smap: dict[str, str], sql: str):
        key = (str(db_path), sql)
        if key in cache:
            return cache[key]
        t = parse_one(sql)
        if t is None:
            cache[key] = (False, False, False)
            return cache[key]
        tabs = base_tables(t, smap)
        all_nonempty = bool(tabs)
        for tb in tabs:
            tk = (str(db_path), 'TBL:' + tb)
            if tk not in cache:
                r = exec_ro(db_path, f'SELECT COUNT(*) FROM "{smap[tb]}"')
                cache[tk] = bool(r and r[0][0] > 0)
            if not cache[tk]:
                all_nonempty = False

        deg_all = False
        deg_lit = False
        if all_nonempty and isinstance(t, exp.Select) and t.args.get('joins'):
            r1 = strip_all_where(t)
            if r1:
                res1 = exec_ro(db_path, f'SELECT COUNT(*) FROM ({r1})')
                if res1 is not None and res1[0][0] == 0:
                    deg_all = True
            r2 = strip_literal_atoms(t)
            if r2:
                res2 = exec_ro(db_path, f'SELECT COUNT(*) FROM ({r2})')
                if res2 is not None and res2[0][0] == 0:
                    deg_lit = True
        if has_irreflexive_self_cmp(t):
            deg_lit = True

        deg_null = False
        for not_node in t.find_all(exp.Not):
            if isinstance(not_node.this, exp.In):
                q = not_node.this.args.get('query')
                sub = q.this if isinstance(q, exp.Subquery) else None
                if isinstance(sub, exp.Select) and len(sub.expressions) == 1 and isinstance(sub.expressions[0], exp.Column):
                    stabs = list(sub.find_all(exp.Table))
                    if len(stabs) == 1 and stabs[0].name.lower() in smap:
                        real_t = smap[stabs[0].name.lower()]
                        col = sub.expressions[0].name
                        r = exec_ro(db_path, f'SELECT COUNT(*) FROM "{real_t}" WHERE "{col}" IS NULL')
                        if r and r[0][0] > 0:
                            deg_null = True

        cache[key] = (deg_all, deg_lit, deg_null)
        return cache[key]

    # 1-3. Pairwise evaluation on 4,017 rival pairs
    pair_stats = {
        '3a_strip_all_where': {'wins': 0, 'losses': 0},
        '3a_strip_literal_atoms': {'wins': 0, 'losses': 0},
        '3a_not_in_null': {'wins': 0, 'losses': 0},
    }
    for pair in pairs_doc['pairs']:
        split = pair['split']
        rmap, manifest, folder = banks[split]
        row = rmap[pair['id']]
        meta = manifest['databases'][row['db_id']]
        db_path = folder / meta['public_path']
        smap = {t['name'].lower(): t['name'] for t in meta['schema']}

        ga, gl, gn = check_3a(db_path, smap, pair['good_sql'])
        ba, bl, bn = check_3a(db_path, smap, pair['bad_sql'])
        for name, gd, bd in [
            ('3a_strip_all_where', ga, ba),
            ('3a_strip_literal_atoms', gl, bl),
            ('3a_not_in_null', gn, bn),
        ]:
            if bd and not gd:
                pair_stats[name]['wins'] += 1
            elif gd and not bd:
                pair_stats[name]['losses'] += 1

    for name, st in pair_stats.items():
        tot = st['wins'] + st['losses']
        st['precision'] = round(st['wins'] / tot, 4) if tot else 0.0
        st['net'] = st['wins'] - st['losses']

    # 4. End-to-end check on 152 real empty-example tasks (146 correct + 6 failed)
    e2e_3a_lit = {'correct_wins': 0, 'correct_losses': 0, 'failed_wins': 0, 'failed_losses': 0, 'triggered': 0}
    e2e_3b_vote = {'correct_wins': 0, 'correct_losses': 0, 'failed_wins': 0, 'failed_losses': 0, 'changed': 0}

    for split in ('train', 'validation'):
        rmap, manifest, folder = banks[split]
        for tid, row in rmap.items():
            if row['example']['row_count'] != 0:
                continue
            p, g = replays[split][tid]
            sub_passed = bool(g['passed'])
            meta = manifest['databases'][row['db_id']]
            db_path = folder / meta['public_path']
            smap = {t['name'].lower(): t['name'] for t in meta['schema']}
            hist = [e for e in p['trace'] if e['tool'] != 'submit']
            passing_evs = [e for e in hist if e['tool'] == 'check' and e['observation']['matches_example']]

            # 3a_strip_literal_atoms end-to-end check
            _, sub_deg, _ = check_3a(db_path, smap, p['sql'])
            if sub_deg and passing_evs and p['tool_calls'] < 8:
                e2e_3a_lit['triggered'] += 1
                non_deg = [e['sql'] for e in passing_evs if e['sql'] != p['sql'] and not check_3a(db_path, smap, e['sql'])[1]]
                if non_deg:
                    new_passed = grade_3inst(row, split, non_deg[0])
                    if new_passed and not sub_passed:
                        e2e_3a_lit['failed_wins'] += 1
                    elif sub_passed and not new_passed:
                        e2e_3a_lit['correct_losses'] += 1

            # 3b_blind_vote_align check across tasks with >=2 distinct passing candidates in trace
            if len({e['sql'] for e in passing_evs}) >= 2:
                raw_schema = next((e['observation'] for e in hist if e['tool'] == 'schema'), None)
                if raw_schema:
                    schema_info = legacy.parse_schema(raw_schema)
                    hints = next((e['observation'] for e in hist if e['tool'] == 'repair_hint'), [])
                    hint_counts = Counter(legacy.normalize_sql(h) for h in hints)
                    sigs = Counter(legacy._table_sig(_parse_sqlite(h)) for h in hints)
                    cands = [
                        dict(sql=e['sql'], source='hint' if e['sql'] in hints else 'check',
                             norm=legacy.normalize_sql(e['sql']), result=e['observation'], order=idx)
                        for idx, e in enumerate(passing_evs)
                    ]
                    obs = {
                        'task': {'id': tid, 'question': row['question']},
                        'issue_sql': row['issue_sql'],
                        'example': row['example'],
                        'ordered': row['ordered'],
                        'remaining': 0,
                        'history': hist,
                    }
                    # Compare baseline score_passing_candidate + 1.5*header vs header_weight=0.0 + aligned_vote_key consensus
                    base_scores = [
                        (score_passing_candidate(c, cands, obs, schema_info, hint_counts, sigs, []) + 1.5 * header_fidelity(c, obs), -c['order'], c['sql'])
                        for c in cands
                    ]
                    vk_counts = Counter(aligned_vote_key(c['sql'], schema_info) for c in cands)
                    new_scores = [
                        (score_passing_candidate(c, cands, obs, schema_info, hint_counts, sigs, [])
                         + 0.75 * (vk_counts[aligned_vote_key(c['sql'], schema_info)] > 1), -c['order'], c['sql'])
                        for c in cands
                    ]
                    base_pick = max(base_scores)[2]
                    new_pick = max(new_scores)[2]
                    if base_pick != new_pick:
                        e2e_3b_vote['changed'] += 1
                        bp = grade_3inst(row, split, base_pick)
                        np = grade_3inst(row, split, new_pick)
                        if np and not bp:
                            if sub_passed:
                                e2e_3b_vote['correct_wins'] += 1
                            else:
                                e2e_3b_vote['failed_wins'] += 1
                        elif bp and not np:
                            if sub_passed:
                                e2e_3b_vote['correct_losses'] += 1
                            else:
                                e2e_3b_vote['failed_losses'] += 1

    out = {
        'elapsed_seconds': round(time.perf_counter() - started, 2),
        'rival_pairs_total': len(pairs_doc['pairs']),
        'pairwise': pair_stats,
        'e2e_3a_strip_literal_atoms': e2e_3a_lit,
        'e2e_3b_blind_vote_align': e2e_3b_vote,
        'gate_passed': bool(
            (pair_stats['3a_strip_literal_atoms']['precision'] >= 0.90
             and pair_stats['3a_strip_literal_atoms']['net'] >= 5
             and e2e_3a_lit['correct_losses'] == 0
             and e2e_3a_lit['failed_wins'] >= 1)
            or (e2e_3b_vote['correct_losses'] == 0 and e2e_3b_vote['failed_wins'] >= 1)
        ),
    }
    write_json(ROOT / 'outputs/plan4/step3_ablation.json', out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
