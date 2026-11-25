"""Diagnose conditions04 on Train (480) and Validation (120) and build the 151-task real empty-example rival-pair benchmark."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
COND = ROOT / 'delivery/goal201/conditions04'
COND_SUB = COND / 'submission'
COND_LIB = COND_SUB / 'lib'
sys.path[:0] = [str(ROOT), str(COND_SUB), str(COND_LIB)]

from mp3_sql.common import MAX_SQL_BYTES, digest, load_bank, write_json
from mp3_sql.environment import Tools
from mp3_sql.score import load_cases
from mp3_sql.sqlite_tool import execute_sql
from mp3_sql.common import result_fingerprint

from parent_agent import sql_key, vote_key
from pool import generate_pool
from extra_repairs import extra_pool

spec = importlib.util.spec_from_file_location('cond_lib_submission', COND_LIB / 'submission.py')
lib_sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib_sub)
parse_schema = lib_sub.parse_schema


def read_json(path: Path):
    return json.loads(path.read_bytes())


_CACHE_LOCK = threading.Lock()


def grade_3inst(row, sql, manifest, cases, case_root, exec_cache):
    metadata = manifest['databases'][row['db_id']]
    tables = [t['name'] for t in metadata['schema']]
    outcomes = []
    for case in cases['tasks'][row['id']]:
        path = case_root / case['path']
        key = (case['database_sha256'], sql, False)
        with _CACHE_LOCK:
            result = exec_cache.get(key)
        if result is None:
            result = execute_sql(path, sql, tables, False)
            with _CACHE_LOCK:
                if len(exec_cache) < 40000:
                    exec_cache[key] = result
        passed = bool(result['ok'] and result_fingerprint(result, row['ordered']) == case['expected'])
        outcomes.append(passed)
    return all(outcomes), outcomes[0]


def analyze_split(split: str, out_dir: Path):
    replay_dir = COND / 'evidence' / f'{split}-replay'
    pred_path = replay_dir / f'{split}.json'
    score_path = replay_dir / f'{split}-score.json'
    doc = read_json(pred_path)
    grade = read_json(score_path)
    assert digest(pred_path) == grade['predictions_sha256'], f'SHA mismatch on {split}'

    rows, manifest, folder = load_bank(split)
    cases, case_root = load_cases(split, folder)
    tools = Tools(folder, manifest, cache={})
    exec_cache = {}
    pool_exec = ThreadPoolExecutor(max_workers=6)

    observed_counts = Counter()
    empty_counts = Counter()
    nonempty_counts = Counter()
    rival_pairs = []
    failed_empty_details = []
    split_records = []

    for idx_row, (row, p, g) in enumerate(zip(rows, doc['predictions'], grade['per_question'], strict=True)):
        assert row['id'] == p['id'] == g['id']
        sub_passed, sub_pub = grade_3inst(row, p['sql'], manifest, cases, case_root, exec_cache)
        assert sub_passed == g['passed'], f"3-inst parity mismatch for {row['id']}: {sub_passed} vs {g['passed']}"

        is_empty = (row['example']['row_count'] == 0)
        # Grade distinct public-passing checked SQLs in observed trace
        trace_passing = {}
        trace_passing[p['sql']] = (sub_passed, sub_pub)
        for ev in p['trace']:
            if ev['tool'] == 'check' and ev['observation']['matches_example']:
                s = ev['sql']
                if s not in trace_passing:
                    trace_passing[s] = grade_3inst(row, s, manifest, cases, case_root, exec_cache)

        have_3inst = any(ok for ok, _ in trace_passing.values())
        if sub_passed:
            bucket = 'correct'
        elif have_3inst:
            bucket = 'selection_available'
        elif sub_pub or len(trace_passing) > 1:
            bucket = 'public_only'
        else:
            bucket = 'no_public_match'

        observed_counts[bucket] += 1
        (empty_counts if is_empty else nonempty_counts)[bucket] += 1

        # Build expanded candidate pool for all 151 real empty-example tasks
        if is_empty:
            raw_schema = tools.schema(row)
            schema_info = parse_schema(raw_schema)
            hints = tools.repair_hint(row) if row['id'] in tools.hints else []
            checked_cands = [
                dict(
                    sql=ev['sql'],
                    source='issue' if ev['sql'] == row['issue_sql'] else 'check',
                    norm=lib_sub.normalize_sql(ev['sql']),
                    result=ev['observation'],
                    order=idx,
                )
                for idx, ev in enumerate(p['trace'])
                if ev['tool'] == 'check'
            ]
            obs0 = {
                'task': {'id': row['id'], 'question': row['question']},
                'issue_sql': row['issue_sql'],
                'example': row['example'],
                'ordered': row['ordered'],
                'remaining': max(0, 8 - p['tool_calls']),
                'history': [ev for ev in p['trace'] if ev['tool'] != 'submit'],
            }

            raw_candidates = []
            raw_candidates.append((p['sql'], 'submitted'))
            for ev in p['trace']:
                if ev['tool'] == 'check':
                    raw_candidates.append((ev['sql'], 'trace_check'))
            raw_candidates.append((row['issue_sql'], 'issue_sql'))
            for h in hints:
                raw_candidates.append((h, 'repair_hint'))
            try:
                for item in generate_pool(obs0, schema_info, hints, pool_cap=80):
                    raw_candidates.append((item['sql'], f"pool:{item.get('source', 'gen')}"))
            except Exception:
                pass
            try:
                for item in extra_pool(obs0, schema_info, hints, checked_cands):
                    raw_candidates.append((item['sql'], f"extra:{item.get('source', 'ext')}"))
            except Exception:
                pass

            dedup = {}
            for sql, src in raw_candidates:
                if not isinstance(sql, str) or not sql.strip():
                    continue
                if len(sql.encode('utf-8')) > MAX_SQL_BYTES:
                    continue
                k = sql_key(sql)
                if k not in dedup:
                    dedup[k] = (sql, [src])
                else:
                    if src not in dedup[k][1]:
                        dedup[k][1].append(src)
                if len(dedup) >= 30:
                    break

            trace_obs = {
                ev['sql']: ev['observation']
                for ev in p['trace']
                if ev['tool'] == 'check'
            }

            def eval_cand(item):
                k, (sql, srcs) = item
                fb = trace_obs.get(sql)
                if fb is None:
                    metadata = manifest['databases'][row['db_id']]
                    tables = [t['name'] for t in metadata['schema']]
                    pub_key = (metadata['public_sha256'], sql, False)
                    with _CACHE_LOCK:
                        res = exec_cache.get(pub_key)
                    if res is None:
                        res = execute_sql(folder / metadata['public_path'], sql, tables, False)
                        with _CACHE_LOCK:
                            if len(exec_cache) < 40000:
                                exec_cache[pub_key] = res
                    matches = bool(res['ok'] and result_fingerprint(res, row['ordered']) == tools.expected[row['id']])
                    fb = {'ok': res['ok'], 'matches_example': matches, 'row_count': len(res['rows'])}
                if fb.get('ok') and fb.get('matches_example') and fb.get('row_count', -1) == 0:
                    if sql in trace_passing:
                        p3, p1 = trace_passing[sql]
                    else:
                        p3, p1 = grade_3inst(row, sql, manifest, cases, case_root, exec_cache)
                    return {
                        'sql': sql,
                        'sql_key': k,
                        'vote_key': vote_key(sql),
                        'sources': srcs,
                        'passed_3inst': p3,
                        'passed_pub': p1,
                    }
                return None

            pub_passing_entries = [
                ent for ent in pool_exec.map(eval_cand, list(dedup.items())) if ent is not None
            ]

            goods = [e for e in pub_passing_entries if e['passed_3inst']]
            bads = [e for e in pub_passing_entries if not e['passed_3inst']]
            seen_pair_keys = set()
            for g_ent in goods:
                for b_ent in bads:
                    pk = (g_ent['sql_key'], b_ent['sql_key'])
                    if g_ent['sql_key'] != b_ent['sql_key'] and pk not in seen_pair_keys:
                        seen_pair_keys.add(pk)
                        rival_pairs.append({
                            'id': row['id'],
                            'split': split,
                            'db_id': row['db_id'],
                            'question': row['question'],
                            'good_sql': g_ent['sql'],
                            'bad_sql': b_ent['sql'],
                            'good_sources': g_ent['sources'],
                            'bad_sources': b_ent['sources'],
                        })

            split_records.append({
                'id': row['id'],
                'db_id': row['db_id'],
                'sub_passed': sub_passed,
                'pub_passing_count': len(pub_passing_entries),
                'goods_count': len(goods),
                'bads_count': len(bads),
            })

            if not sub_passed:
                failed_empty_details.append({
                    'id': row['id'],
                    'split': split,
                    'db_id': row['db_id'],
                    'question': row['question'],
                    'issue_sql': row['issue_sql'],
                    'selected_sql': p['sql'],
                    'tool_calls': p['tool_calls'],
                    'selected_passed_public': sub_pub,
                    'bucket': bucket,
                    'correct_in_trace': have_3inst,
                    'correct_in_expanded_pool': len(goods) > 0,
                    'public_passing_candidates': pub_passing_entries,
                })
        if (idx_row + 1) % 60 == 0:
            print(f'[diagnose4:{split}] {idx_row + 1}/{len(rows)}', flush=True)

    pool_exec.shutdown(wait=True)
    split_report = {
        'split': split,
        'total': len(rows),
        'correct': int(grade['correct']),
        'observed_buckets': dict(observed_counts),
        'empty_total': sum(empty_counts.values()),
        'empty_buckets': dict(empty_counts),
        'nonempty_total': sum(nonempty_counts.values()),
        'nonempty_buckets': dict(nonempty_counts),
        'empty_rival_pairs': len(rival_pairs),
        'empty_tasks_with_rival_pairs': len({r['id'] for r in rival_pairs}),
        'failed_empty_details': failed_empty_details,
        'empty_records': split_records,
    }
    write_json(out_dir / f'diagnose4_{split}.json', split_report)
    return split_report, rival_pairs


def main():
    started = time.perf_counter()
    out_dir = ROOT / 'outputs/plan4'
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rep, train_pairs = analyze_split('train', out_dir)
    val_rep, val_pairs = analyze_split('validation', out_dir)
    all_pairs = train_pairs + val_pairs
    write_json(out_dir / 'empty_rival_pairs.json', {
        'total_pairs': len(all_pairs),
        'total_tasks_with_pairs': len({p['id'] for p in all_pairs}),
        'train_pairs': len(train_pairs),
        'validation_pairs': len(val_pairs),
        'pairs': all_pairs,
    })
    summary = {
        'elapsed_seconds': round(time.perf_counter() - started, 2),
        'train': {k: v for k, v in train_rep.items() if k != 'empty_records'},
        'validation': {k: v for k, v in val_rep.items() if k != 'empty_records'},
        'combined_empty_total': train_rep['empty_total'] + val_rep['empty_total'],
        'combined_empty_correct': train_rep['empty_buckets'].get('correct', 0) + val_rep['empty_buckets'].get('correct', 0),
        'combined_empty_failed': len(train_rep['failed_empty_details']) + len(val_rep['failed_empty_details']),
        'combined_empty_rival_pairs': len(all_pairs),
        'combined_empty_tasks_with_pairs': len({p['id'] for p in all_pairs}),
    }
    write_json(out_dir / 'diagnose4_summary.json', summary)
    print(json.dumps({
        'elapsed_seconds': summary['elapsed_seconds'],
        'train_correct': f"{train_rep['correct']}/{train_rep['total']}",
        'train_buckets': train_rep['observed_buckets'],
        'train_empty_buckets': train_rep['empty_buckets'],
        'val_correct': f"{val_rep['correct']}/{val_rep['total']}",
        'val_buckets': val_rep['observed_buckets'],
        'val_empty_buckets': val_rep['empty_buckets'],
        'combined_empty_correct': f"{summary['combined_empty_correct']}/{summary['combined_empty_total']}",
        'combined_empty_rival_pairs': summary['combined_empty_rival_pairs'],
        'combined_empty_tasks_with_pairs': summary['combined_empty_tasks_with_pairs'],
        'failed_empty_ids': [d['id'] for d in train_rep['failed_empty_details'] + val_rep['failed_empty_details']],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
