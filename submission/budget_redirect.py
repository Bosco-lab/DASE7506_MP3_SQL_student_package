"""Alternative hypotheses after a public failure. Return SQL for one normal check."""
import re
from sqlglot import exp
from extra_repairs import _headers, _parse, _set, _sql, _tables
from local_edits import shortest


def alternatives(obs, info, hints):
    example = obs['example']
    headers = example['columns']
    question = obs['task']['question'].lower()
    checked = {e['sql']: e['observation'] for e in obs['history'] if e['tool'] == 'check'}
    for seed in dict.fromkeys([*hints, obs['issue_sql']]):
        tree = _parse(seed)
        if not isinstance(tree, exp.Select):
            continue
        tables = _tables(tree)
        projected = _headers(headers, tables, info['tables'], tree.expressions)
        having = tree.args.get('having')
        # Grouped minimum/maximum with a nested count. Keep the question's
        # grouping, rank groups by size, and project just the requested fields.
        if tree.args.get('group') and having and having.find(exp.Subquery) and tree.find(exp.Count) and projected:
            low = re.search(r'\b(least|fewest|smallest|lowest)\b', question)
            high = re.search(r'\b(most|largest|highest)\b', question)
            if low or high:
                ranked = _set(tree, expressions=projected[0], having=None, order=None, limit=None)
                yield 5, _sql(ranked.order_by(exp.Ordered(this=exp.Count(this=exp.Star()), desc=bool(high))).limit(1))
        if tree.find(exp.Subquery, exp.CTE) or not tables:
            continue
        # Rebind the projection *and the rest of the clauses* to one visible
        # relation. Useful when the draft joined a label table redundantly.
        if len(tables) > 1:
            for t in tables:
                meta = info['tables'].get(t.name.lower())
                shapes = _headers(headers, [t], info['tables'], tree.expressions)
                if not meta or not shapes:
                    continue
                node = _set(tree, expressions=shapes[0], joins=None).from_(t.copy())
                valid = True
                for c in node.find_all(exp.Column):
                    name = c.name.lower()
                    if name not in meta['columns']:
                        valid = False
                        break
                    c.set('table', exp.to_identifier(t.alias_or_name, quoted=True))
                if valid:
                    yield 3, _sql(node)
        # The example can distinguish an aggregate over all rows from a list
        # of individual values. Preserve the original filter and ordering.
        if len(tables)==1 and len(headers)==1 and not tree.args.get('group') and example['row_count']>1 and tree.find(exp.AggFunc) and projected:
            plain = projected[0][0]
            if isinstance(plain, exp.Column):
                node = _set(tree, expressions=[plain], order=None)
                if re.search(r'\b(ascending|descending)\b', question):
                    node = node.order_by(exp.Ordered(this=plain.copy(), desc='descending' in question))
                yield 4, _sql(node)
        # Join multiplicity and DISTINCT only affect duplicates. Compare the
        # visible rows when complete before allocating a check to this change.
        result = checked.get(seed)
        if not result or not result['ok'] or not result['rows'] or not example['rows'] or result['truncated'] or example['truncated']:
            continue
        rows = list(map(tuple, result['rows']))
        expected = list(map(tuple, example['rows']))
        if set(rows)==set(expected) and len(rows)!=len(expected):
            yield 6, _sql(_set(tree, distinct=None if tree.args.get('distinct') else exp.Distinct()))


def bind_missing(tree, info):
    """A wrong qualifier can still name a real field one FK away. Repair its owner."""
    tables = _tables(tree)
    if not tables or tree.find(exp.Subquery, exp.CTE):
        return
    schema = info['tables']
    aliases = {t.alias_or_name.lower():t for t in tables}
    for col in tree.find_all(exp.Column):
        ref = aliases.get(col.table.lower())
        name = col.name.lower()
        owners = info['col_to_tables'].get(name, [])
        if not ref or len(owners) != 1 or name in schema.get(ref.name.lower(), {}).get('columns', {}):
            continue
        target = owners[0].lower()
        dest = next((t for t in tables if t.name.lower()==target), None)
        joins = list(tree.args.get('joins') or [])
        if dest is None:
            edges = shortest(info, ref.name.lower(), target)
            if len(edges) != 1:
                continue
            edge = edges[0]
            dest = exp.to_table(owners[0], quoted=True)
            on = exp.EQ(this=exp.column(edge['from_col'], table=ref.alias_or_name, quoted=True),
                        expression=exp.column(edge['to_col'], table=dest.alias_or_name, quoted=True))
            joins.append(exp.Join(this=dest.copy(), on=on))
        fixed = _set(tree, joins=joins)
        for c in fixed.find_all(exp.Column):
            if c.table.lower()==col.table.lower() and c.name.lower()==name:
                c.set('table', exp.to_identifier(dest.alias_or_name, quoted=True))
        yield _sql(fixed)
