"""P2 candidate generation. No tool calls, labels or controller state."""
from collections import Counter
from difflib import SequenceMatcher
import re

from sqlglot import exp

# Match the agent's sibling-module loading for both packaged and flat imports.
if __package__:
    from pathlib import Path
    import sys
    directory = str(Path(__file__).resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)

import ast_pool as broad
import ast_repair as repair

# First canonical producer owns the candidate (also for ablations). Static
# parents remain available for second-order generation when their output is pruned.
OLD = ('seed', 'proj', 'qual', 'simp', 'name', 'tmpl', 'agg', 'basic', 'simp2')
NEW = ('hint4', 'schema_fix', 'align', 'fk_join')
DYNAMIC = ('feedback_error', 'feedback_projection', 'feedback_semantic')
RETAINED = frozenset(('seed', 'simp', 'simp2', 'tmpl', 'align', 'fk_join', 'schema_fix'))


def _fk_candidates(tree, schema):
    refs = repair._table_alias_map(tree, schema)
    active = {ref['table'].lower() for ref in refs}
    aliases = {ref['alias'].lower(): ref['table'].lower() for ref in refs}
    for col in repair._direct_columns_in_select(tree):
        owners = schema['col_to_tables'].get(col.name.lower(), [])
        if not col.table and any(o.lower() in active for o in owners):
            continue
        if aliases.get(col.table.lower()) in {o.lower() for o in owners}:
            continue
        for owner in owners:
            if owner.lower() in active:
                continue
            paths = [repair._shortest_fk_path([t], owner, schema) for t in sorted(active)]
            path = min((p for p in paths if p), key=len, default=None)
            if path and len(path) <= 2:
                cloned = tree.copy()
                alias = repair._insert_fk_joins(cloned, refs, path)
                for c in repair._direct_columns_in_select(cloned):
                    if c.name.lower() == col.name.lower() and c.table.lower() == col.table.lower():
                        c.set('table', exp.to_identifier(alias))
                yield repair._safe_sql(cloned)


def _column_fixes(tree, schema, headers):
    refs = repair._table_alias_map(tree, schema)
    header_names = {repair._norm_ident(h) for h in headers}
    for col in repair._direct_columns_in_select(tree):
        if not col.name or col.name == '*':
            continue
        local = [r for r in refs if not col.table or col.table.lower() == r['alias'].lower()]
        matches = [r for r in local if repair._col_exists_in_table(r['table'], col.name, schema)]
        if len(matches) > 1 and not col.table:
            for r in matches:
                cloned = tree.copy()
                for c in repair._direct_columns_in_select(cloned):
                    if not c.table and c.name.lower() == col.name.lower():
                        c.set('table', exp.to_identifier(r['alias']))
                yield repair._safe_sql(cloned)
        if matches:
            continue
        choices = []
        for r in refs:
            for name in schema['tables'].get(r['table'].lower(), {}).get('columns', {}).values():
                if r not in local and name.lower() != col.name.lower():
                    continue
                score = SequenceMatcher(None, name.lower(), col.name.lower()).ratio()
                if repair._norm_ident(name) in header_names:
                    score += .15
                if col.name.lower() in name.lower() or name.lower() in col.name.lower():
                    score += .25
                if score >= .62:
                    choices.append((score, name, r['alias']))
        for _, name, alias in sorted(choices, reverse=True)[:2]:
            cloned = tree.copy()
            for c in repair._direct_columns_in_select(cloned):
                if (c.table, c.name) == (col.table, col.name):
                    c.set('this', exp.to_identifier(name))
                    c.set('table', exp.to_identifier(alias))
            yield repair._safe_sql(cloned)


def _catalog(obs, schema, hints, feedback=None):
    """Uncapped evaluation catalog in stable priority order, incl. fourth draw."""
    hints = [hints] if isinstance(hints, str) else list(hints)
    votes = Counter(broad.canonical_sql(h) for h in hints)
    seeds = [(f'hint{i}', sql) for i, sql in enumerate(hints)]
    seeds.insert(min(1, len(seeds)), ('issue', obs['issue_sql']))
    pool, seen = [], set()

    def add(sql, family, base, source=None):
        if not sql or not broad._is_valid_pool_sql(sql, schema):
            return
        key = broad.canonical_sql(sql)
        if key not in seen:
            seen.add(key)
            pool.append(dict(source=source or f'pool_{family}_{base}', family=family,
                             base=base, canonical=key, sql=sql, hint_votes=votes[key]))

    for base, sql in seeds:
        source = 'hint_alt' if base.startswith('hint') and base != 'hint0' else base
        add(sql, 'hint4' if base == 'hint3' else 'seed', base, source)

    question = obs['task']['question']
    for base, sql in seeds[:2]:
        t = repair._parse_sqlite(sql)
        if not isinstance(t, exp.Select):
            continue
        for family, candidates in (
            ('proj', broad._column_guided_projections(t, obs['example'], schema)),
            ('qual', broad._unqualify_and_qualify_variants(t, schema)),
            ('simp', broad._simplifications(t, schema, question, strip_percent=True)),
            ('name', broad._name_variants(t, schema)),
            ('tmpl', broad._split_anti_union_and_intersect(t, question, schema, strip_percent=True)),
            ('agg', broad._aggregate_column_variants(t, schema)),
            ('basic', broad._basic_variants(t, schema)),
        ):
            for candidate in candidates:
                add(candidate, family, base)
    for parent in pool[:20]:
        t = repair._parse_sqlite(parent['sql'])
        if isinstance(t, exp.Select):
            for sql in broad._simplifications(t, schema, question, strip_percent=True)[:4]:
                add(sql, 'simp2', parent['base'])

    for base, sql in seeds:
        t = repair._parse_sqlite(sql)
        if not isinstance(t, exp.Select):
            continue
        ok, fixed = repair.validate_and_fix_schema_columns(sql, schema, obs['example'])
        if ok:
            add(fixed, 'schema_fix', base)
        for fixed in _column_fixes(t, schema, obs['example']['columns']):
            add(fixed, 'schema_fix', base)
        for candidate in _fk_candidates(t, schema):
            add(candidate, 'fk_join', base)
        # Header-only alignment path: no executed rows, no row-match scoring.
        headers = [e.alias_or_name or e.sql(dialect='sqlite') for e in t.expressions]
        headers_only = dict(sql=sql, result=dict(ok=True, columns=headers, rows=[]))
        for candidate in repair._projection_repairs(headers_only, obs, schema):
            add(candidate, 'align', base)
        # If each requested header uniquely resolves, try the full ordered
        # schema projection, including same-arity wrong-column drafts.
        refs = repair._table_alias_map(t, schema)
        projected = []
        for hdr in obs['example']['columns']:
            agg = re.fullmatch(r'(count|sum|avg|max|min)\s*\(\s*(\*|\w+)\s*\)', hdr, re.I)
            if agg and agg[0].lower().startswith('count') and agg[2] == '*':
                projected.append(exp.Count(this=exp.Star()))
                continue
            column = agg[2] if agg else hdr
            owners = [(r['alias'], repair._col_exists_in_table(r['table'], column, schema)) for r in refs]
            hits = [(a, c) for a, c in owners if c]
            if len(hits) != 1:
                break
            alias, name = hits[0]
            expr = exp.column(name, table=alias)
            projected.append(exp.func(agg[1], expr) if agg else expr)
        if projected and len(projected) == len(obs['example']['columns']):
            cloned = t.copy()
            cloned.set('expressions', projected)
            add(repair._safe_sql(cloned), 'align', base)

    for i, event in enumerate(feedback or ()):
        if event['tool'] != 'check':
            continue
        base = next((b for b, s in seeds if s == event['sql']), f'check{i}')
        cand = dict(sql=event['sql'], result=event['observation'], source=base, order=i)
        for family, candidates in (
            ('feedback_error', repair._exec_error_repairs(cand, obs, schema)),
            ('feedback_projection', repair._projection_repairs(cand, obs, schema)),
            ('feedback_semantic', repair._semantic_and_spurious_variants([cand], obs, schema)),
        ):
            for candidate in candidates:
                add(candidate, family, base)
                if family == 'feedback_error':
                    # A missing-column fix can still leave extra SELECT items.
                    # Align that repaired AST without assuming its result rows.
                    fixed = repair._parse_sqlite(candidate)
                    if isinstance(fixed, exp.Select):
                        for aligned in broad._column_guided_projections(fixed, obs['example'], schema):
                            add(aligned, family, base)
    return pool


def _select(catalog, families, cap=80):
    selected = [c for c in catalog if c['family'] in families][:max(0, cap)]
    return [dict(c, rank_in_pool=i) for i, c in enumerate(selected)]


def generate_pool(obs, schema, hints, feedback=None, pool_cap=80):
    """Return exact SQL and zero-based rank. hints are the raw repair_hint draws."""
    history = feedback if RETAINED.intersection(DYNAMIC) else None
    return _select(_catalog(obs, schema, hints, history), RETAINED, pool_cap)
