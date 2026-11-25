"""Schedule a single real check per decision, using only current episode observations."""
from dataclasses import dataclass

from sqlglot import exp
from ast_repair import _parse_sqlite, validate_and_fix_schema_columns
from extra_repairs import _headers, _set, _sql, _tables
from local_edits import grounded_join
from budget_redirect import alternatives


@dataclass(frozen=True)
class Candidate:
    key: str
    screen: str = ''


class CheckBudget:
    def __init__(self, mode):
        self.mode = mode
        self.cache = {}
        self.ready = False
        self.first_ready = False

    def describe(self, sql, info, example):
        if sql in self.cache:
            return self.cache[sql]
        tree = _parse_sqlite(sql)
        item = self._describe(sql, tree, info, example)
        self.cache[sql] = item
        return item

    @staticmethod
    def _describe(sql, tree, info, example):
        if not isinstance(tree, (exp.Select, exp.SetOperation)):
            return Candidate(sql, 'parse')
        if tree.find(exp.All, exp.Any):
            return Candidate(sql, 'quantified comparison')
        if isinstance(tree, exp.SetOperation):
            return Candidate(_sql(tree))
        known_width = all(not e.is_star for e in tree.expressions)
        wrong_width = known_width and len(tree.expressions) != len(example['columns'])
        reason = 'width' if wrong_width else ''
        # Alpha conversion only in a single flat scope with known column ownership.
        if sum(1 for _ in tree.find_all(exp.Select)) != 1:
            return Candidate(_sql(tree), reason)
        sources = _tables(tree)
        if not sources or tree.find(exp.CTE, exp.Subquery) or len(sources) != 1 + len(tree.args.get('joins') or []):
            return Candidate(_sql(tree), reason)
        columns = {t.alias_or_name.lower(): info['tables'].get(t.name.lower(), {}).get('columns', {}) for t in sources}
        if not all(columns.values()):
            return Candidate(_sql(tree), 'table')
        aliases = {e.alias.lower() for e in tree.expressions if e.alias}
        table_names = {t.alias_or_name.lower(): t.name.lower() for t in sources}
        # Refuse to collapse self joins or resolve aliases shadowing other names.
        simple = len(set(table_names.values())) == len(sources) == len(columns)
        used = set()
        for c in tree.find_all(exp.Column):
            name, qual = c.name.lower(), c.table.lower()
            if name == '*':
                if qual: simple = False
                continue
            if not qual and name in aliases:
                used.add(name)
                continue
            owners = [alias for alias, cols in columns.items() if name in cols]
            pseudo = name in {'rowid','_rowid_','oid','true','false'} or c.this.args.get('quoted')
            if qual and (qual not in columns or name not in columns[qual]) or not qual and not owners and not pseudo:
                return Candidate(_sql(tree), 'column')
            if len(owners) == 1 and simple:
                c.set('table', exp.to_identifier(table_names[owners[0]], quoted=True))
                c.set('this', exp.to_identifier(name, quoted=True))
            elif qual and simple and qual in table_names:
                c.set('table', exp.to_identifier(table_names[qual], quoted=True))
        if simple:
            for table in sources:
                table.set('alias', None)
                table.set('this', exp.to_identifier(table.name.lower(), quoted=True))
        for e in list(tree.expressions):
            if isinstance(e, exp.Alias) and e.alias.lower() not in used:
                e.replace(e.this.copy())
        # DISTINCT of one scalar aggregate row cannot affect any row or multiplicity.
        if not tree.args.get('group') and all(isinstance(e, exp.AggFunc) for e in tree.expressions):
            tree.set('distinct', None)
        return Candidate(_sql(tree), reason)

    def eligible(self, sql, info, obs, checked):
        item = self.describe(sql, info, obs['example'])
        return not item.screen and item.key not in checked

    def seen(self, obs, info):
        return {self.describe(e['sql'], info, obs['example']).key for e in obs['history'] if e['tool'] == 'check'}

    def next_hint(self, agent, obs):
        """Issue -> hints -> schema -> representatives. Raw hint votes remain available."""
        self.first_ready = False
        if self.ready or not obs['remaining']:
            return None
        history = obs['history']
        if not history:
            return agent._check(obs['issue_sql'], 'issue')
        agent._ingest_history(history, obs['issue_sql'])
        if not agent.hints_fetched:
            return {'tool': 'repair_hint'}
        if not agent.schema_fetched:
            return {'tool': 'schema'}
        info = agent.schema_info
        seen = self.seen(obs, info)
        exact = {e['sql'] for e in history if e['tool'] == 'check'}
        from parent_agent import sql_key
        exact_keys = {sql_key(s) for s in exact}
        hints = list(dict.fromkeys(agent.raw_hint_list))
        proposals = []
        for sql in hints:
            if sql_key(sql) in exact_keys:
                continue
            if self.mode == 'schema' or self.eligible(sql, info, obs, seen):
                proposals.append((2, sql, 'hint'))
        failed = not any(e['observation']['matches_example'] for e in history if e['tool'] == 'check')
        if self.mode in ('redirect','feedback','binding') and failed and not proposals:
            # Invalid drafts are still seeds. Bind their requested projection or fix
            # an FK edge before checking. Favor complete, executable hint hypotheses.
            for sql in hints + [obs['issue_sql']]:
                item = self.describe(sql, info, obs['example'])
                if item.screen in {'column', 'width'}:
                    tree = _parse_sqlite(sql)
                    if not isinstance(tree, exp.Select):
                        continue
                    edits = []
                    if item.screen == 'column':
                        ok, fixed = validate_and_fix_schema_columns(sql, info, obs['example'])
                        if ok: edits.append(fixed)
                        edits.extend(_sql(node) for _, _, node in grounded_join(tree, info))
                        if self.mode == 'binding':
                            from budget_redirect import bind_missing
                            edits.extend(bind_missing(tree, info))
                    edits.extend(_sql(_set(tree, expressions=p)) for p in _headers(obs['example']['columns'], _tables(tree), info['tables'], tree.expressions)[:2])
                    proposals.extend((1, s, 'budget_fix') for s in edits if self.eligible(s, info, obs, seen))
        if self.mode in ('feedback','binding') and failed and not any(p[2]=='hint' for p in proposals):
            if sum(agent.sources.get(s)=='budget_route' for s in exact) < 2:
                proposals.extend((priority, sql, 'budget_route') for priority,sql in alternatives(obs, info, hints)
                                 if self.eligible(sql, info, obs, seen))
        if proposals:
            _, sql, source = max(proposals, key=lambda x: x[0])
            return agent._check(sql, source)
        self.ready = self.first_ready = True
        return None
