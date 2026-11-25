"""Hybrid SQL repair agent. Runtime inputs are the current episode and frozen weights."""
from collections import Counter
import re
from sqlglot import exp
from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LIB = HERE / 'lib' if (HERE / 'lib' / 'submission.py').is_file() else HERE.parent / 'submission'
V5 = HERE.parent / 'submission_v5'
sys.path[:0] = [str(HERE), str(LIB), str(V5)]
from ast_repair import (detect_spurious_wakeup_defect, generate_ast_repairs, generate_probe_repairs,
                        propose_value_probe, rewrite_sql_with_probes, _parse_sqlite, _safe_sql)
from pool import generate_pool
from v5_rank import features, score
from ast_repair import score_passing_candidate, validate_and_fix_schema_columns
from ast_pool import _is_valid_pool_sql, build_broad_ast_pool

spec = importlib.util.spec_from_file_location('goal_legacy_helpers', LIB / 'submission.py')
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


@lru_cache(maxsize=512)
def sql_key(sql):
    # Keep case and spaces inside string literals (important after a value probe).
    tree = _parse_sqlite(sql)
    return (_safe_sql(tree) if tree is not None else None) or sql.strip()


@lru_cache(maxsize=12)
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def weights_path():
    return next(p / 'weights.json' for p in (HERE, LIB, V5) if (p / 'weights.json').is_file())


@lru_cache(maxsize=512)
def vote_key(sql):
    """Ignore local alias spelling, retaining predicate structure and literal case."""
    tree = _parse_sqlite(sql)
    if not isinstance(tree, exp.Select) or tree.find(exp.Subquery, exp.CTE):
        return sql_key(sql)
    tree = tree.copy()
    tables = list(tree.find_all(exp.Table))
    if len({t.name.casefold() for t in tables}) != len(tables):
        return sql_key(sql)
    aliases = {t.alias_or_name.casefold(): t.name.casefold() for t in tables}
    for col in tree.find_all(exp.Column):
        if col.table.casefold() in aliases:
            col.set('table', exp.to_identifier(aliases[col.table.casefold()]) if len(tables)>1 else None)
    for table in tables:
        table.set('alias', None)
    return tree.sql(dialect='sqlite', normalize=True)


def header_fidelity(candidate, obs):
    want = obs['example']['columns']
    norm = lambda s: re.sub(r'\s+', '', s.casefold()).split('.')[-1]
    return sum(norm(a)==norm(b) for a,b in zip(candidate['result']['columns'], want)) / max(1, len(want))


class Agent(legacy.SQLRepairAgent):
    def __init__(self, cfg, priority_model):
        super().__init__({'enable_spurious_guard': True, 'enable_learned_ranker': False})
        self.cfg = cfg
        self.model = priority_model
        self.pool_scores = None
        self.completions = 0
        self.sources = {}
        self.extra = None
        self.independent_votes = set()
        self.jury_queue = []
        self.jury_risk = None
        if cfg['enable_extra']:
            from extra_repairs import extra_pool
            self.extra = extra_pool

    def _check(self, sql, source):
        self.sources[sql] = source
        return {'tool': 'check', 'sql': sql}

    def worth_check(self, sql, obs):
        policy = self.budget
        if policy.mode == 'schema' or not self.schema_info:
            return True
        return policy.eligible(sql, self.schema_info, obs, policy.seen(obs, self.schema_info))

    def _ranked(self, obs):
        if self.pool_scores is None:
            pool = generate_pool(obs, self.schema_info, self.raw_hint_list, pool_cap=80)
            self.pool_scores = {sql_key(c['sql']): (score(self.model, features(obs, self.schema_info,
                                self.raw_hint_list, c)), c['sql'], c['source']) for c in pool}
        return self.pool_scores

    def _llm_check(self, obs, done, passing):
        def next_beam():
            while self.llm_beam_queue:
                sql = self.llm_beam_queue.pop(0)
                if sql_key(sql) not in done:
                    return self._check(sql, 'llm_lora')
            return None

        pending = next_beam()
        if pending or self.completions == 6:
            return pending
        if not self.retrieved and obs['remaining'] >= 3:
            return {'tool': 'retrieve', 'query': legacy.build_retrieve_query(obs['task']['question'], obs['issue_sql'])}
        from llm_refiner import propose_llm_beams
        count = min(3, 6 - self.completions)
        self.completions += count  # charge every returned sequence, including invalid/duplicate beams
        self.llm_rounds += 1
        beams = propose_llm_beams(obs, self.schema_info, self.retrieved_examples, self.candidates,
                                  use_lora=True, num_beams=count, round_idx=self.llm_rounds,
                                  probed_values=self.probe_results)
        for beam in beams:
            sql = beam['sql']
            sql = rewrite_sql_with_probes(sql, self.schema_info, self.probe_results) or sql
            if passing and (not beam['schema_valid'] or detect_spurious_wakeup_defect(
                    sql, obs, self.schema_info, [{'sql': sql}])):
                continue
            self.llm_beam_queue.append(sql)
        return next_beam()

    def _risk(self, obs, passing, best):
        rows = obs['example']['row_count']
        tree = _parse_sqlite(best['sql'])
        if rows > 1 or tree is None:
            return -1.
        hints = [vote_key(h) for h in self.raw_hint_list]
        agreement = hints.count(vote_key(best['sql'])) / max(1, len(hints))
        joins = len(list(tree.find_all(exp.Join)))
        agg_q = re.search(r'\b(how many|number of|average|total|most common|least common)\b', obs['task']['question'], re.I)
        mismatch = bool(agg_q and not tree.find(exp.AggFunc))
        disagreement = len({vote_key(c['sql']) for c in passing}) > 1
        defect = bool(detect_spurious_wakeup_defect(best['sql'], obs, self.schema_info, passing))
        return (1.5*(rows == 0) + .55*min(joins, 3) + 1.3*(1-agreement) + disagreement
                + (1-header_fidelity(best, obs)) + mismatch + defect)

    def _vote_terms(self, obs, passing):
        signatures = Counter(legacy._table_sig(_parse_sqlite(h)) for h in self.raw_hint_list)
        return [(c['sql'], score_passing_candidate(c, passing, obs, self.schema_info,
                        self.raw_hint_counts, signatures, self.probe_results),
                 int(vote_key(c['sql']) in self.independent_votes), header_fidelity(c, obs)) for c in passing]

    def _unseen(self, obs, done):
        options = [(2.-.2*i, sql) for i, sql in enumerate(self.llm_beam_queue)]
        options.extend((4., s) for s in self.raw_hint_list)
        if self.schema_info:
            options.extend((logit, s) for logit,s,_ in self._ranked(obs).values())
            if self.extra:
                options.extend((10*(c['priority']-.76), c['sql']) for c in self.extra(
                    obs, self.schema_info, self.raw_hint_list, self.candidates))
            options.extend((1.2-.15*i, s) for i,(s,_) in enumerate(generate_ast_repairs(
                obs, self.schema_info, self.candidates, done, sql_key)))
            broad = build_broad_ast_pool(obs, self.schema_info, self.raw_hint_list, pool_cap=80)
            options.extend((-5.-i*.01, c['sql']) for i,c in enumerate(broad))
            # Exhausted repair pools: try an ordered projection of the requested columns.
            for table in self.schema_info['tables'].values():
                cols = [table['columns'].get(h.casefold()) for h in obs['example']['columns']]
                if cols and all(cols):
                    options.append((-8., exp.select(*(exp.column(c, quoted=True) for c in cols)).from_(
                        exp.to_table(table['name'], quoted=True)).sql(dialect='sqlite')))
        for _, sql in sorted(options, key=lambda x:-x[0]):
            if sql_key(sql) in done or not _is_valid_pool_sql(sql, self.schema_info):
                continue
            valid, fixed = validate_and_fix_schema_columns(sql, self.schema_info, obs['example'])
            if valid and sql_key(fixed) not in done:
                return fixed
        # Every existing candidate can be exhausted. A schema-grounded partial projection
        # still gives a valid new hypothesis. Do not resubmit a previously failed check.
        for table in (self.schema_info or {}).get('tables', {}).values():
            for col in table['columns'].values():
                query = exp.select(exp.column(col, quoted=True)).from_(exp.to_table(table['name'], quoted=True))
                for variant in (query, query.distinct(), query.limit(1)):
                    sql = variant.sql(dialect='sqlite')
                    if sql_key(sql) not in done:
                        return sql
        # With a shortened episode budget schema can still be unavailable. Keep the
        # original SELECT when it has no failed check, otherwise use a fresh read-only SQL.
        if sql_key(obs['issue_sql']) not in done:
            return obs['issue_sql']
        return next(s for i in range(len(done)+1) if sql_key(s := f'SELECT NULL LIMIT {i+1}') not in done)

    def _finish(self, obs, passing, done):
        if not passing:
            return {'tool':'submit', 'sql':self._unseen(obs, done)}
        best_sql = self._select_best_sql(obs)
        if self.jury_risk is None and self.cfg['enable_llm'] and obs['remaining'] and self.completions <= 3:
            best = next(c for c in passing if c['sql']==best_sql)
            risk = self._risk(obs, passing, best)
            if risk >= self.cfg['risk_threshold']:
                from llm_refiner import propose_llm_beams
                self.jury_risk = risk
                self.completions += 3
                self.llm_rounds += 1
                # Blind second opinion. No current drafts, checks, retrieved SQL or probes.
                beams = propose_llm_beams(obs, self.schema_info, [], [], num_beams=3, round_idx=self.llm_rounds)
                for beam in beams:
                    if beam['schema_valid'] and beam['col_count_match']:
                        self.independent_votes.add(vote_key(beam['sql']))
                        self.jury_queue.append(beam['sql'])
        if self.jury_risk is not None:
            while obs['remaining'] and self.jury_queue:
                sql = self.jury_queue.pop(0)
                if sql_key(sql) not in done:
                    return self._check(sql, 'independent')
            if self.independent_votes:
                terms = self._vote_terms(obs, passing)
                best_sql = max(terms, key=lambda x:x[1]+self.cfg['vote_weight']*x[2]+self.cfg['header_weight']*x[3])[0]
        return {'tool':'submit', 'sql':best_sql}

    def act(self, obs):
        history = obs['history']
        self._ingest_history(history, obs['issue_sql'])
        # Preserve case-only repairs and provenance in both the manual selector and prompts.
        self.candidates = [dict(sql=e['sql'], source=self.sources.get(e['sql'], 'issue' if e['sql'] == obs['issue_sql'] else 'hint'),
                                norm=legacy.normalize_sql(e['sql']), result=e['observation'], order=i)
                           for i, e in enumerate(history) if e['tool'] == 'check']
        rem = obs['remaining']
        done = {sql_key(c['sql']) for c in self.candidates}
        passing = [c for c in self.candidates if c['result']['matches_example']]
        if rem == 0 or self.jury_risk is not None:
            return self._finish(obs, passing, done)
        if not self.candidates:
            return self._check(obs['issue_sql'], 'issue')
        if not self.hints_fetched:
            return {'tool': 'repair_hint'}
        # Outer scheduler has already consumed or screened the raw hints.

        suspicious = passing and detect_spurious_wakeup_defect(self._select_best_sql(obs), obs,
                                                                self.schema_info, passing)
        options = []  # sortable policy utility, SQL, legacy source tag
        if self.extra and (not passing or suspicious):
            for item in self.extra(obs, self.schema_info, self.raw_hint_list, self.candidates):
                if legacy.is_valid_sql_payload(item['sql']):
                    options.append((10 * (item['priority'] - .76), item['sql'], item['source']))
        if self.probe_results:
            options.extend((8., sql, src) for sql, src in generate_probe_repairs(
                self.candidates, self.schema_info, self.probe_results, done, sql_key))
        if self.cfg['feedback']:
            options.extend((self.cfg['feedback_priority'] - .15*i, sql, src)
                           for i, (sql, src) in enumerate(generate_ast_repairs(
                               obs, self.schema_info, self.candidates, done, sql_key)))
        # Easy, unambiguous matches do not require materializing the full static AST pool.
        explore = not passing or (suspicious and len(passing) == 1)
        if explore:
            options.extend(self._ranked(obs).values())
        best = next((x for x in sorted(options, key=lambda x:-x[0])
                     if sql_key(x[1]) not in done and self.worth_check(x[1], obs)), None)
        confident = best and best[0] >= self.cfg['confident']
        if self.cfg['probe'] and not self.probe_done and rem >= 2:
            probe = propose_value_probe(obs, self.schema_info, self.candidates, set())
            if probe:
                sql, table, col = probe
                self.probe_meta_by_norm[legacy.normalize_sql(sql)] = (table, col)
                return {'tool': 'execute', 'sql': sql}

        need_model = self.cfg['enable_llm'] and (not passing or suspicious) and not any(
            c['source'] == 'llm_lora' for c in passing)
        if need_model and (self.llm_rounds or not best or rem <= 2 or (not confident and rem <= 3)):
            action = self._llm_check(obs, done, passing)
            if action:
                return action
        if best and (not passing or suspicious or best[0] >= self.cfg['pass_floor']):
            return self._check(best[1], best[2])
        return self._finish(obs, passing, done)


def create_agent(checkpoint=None):
    cfg = read_json(checkpoint or HERE / 'config.json')
    return Agent(cfg, read_json(weights_path())['priority'])
