"""Recover a projection when observed values demonstrate the column mapping."""
from collections import Counter
from itertools import permutations
from extra_repairs import _parse, _sql, _set
from sqlglot import exp


def projections(obs):
    example=obs['example']
    rows=example['rows']
    if not rows or example['truncated']:
        return
    width=len(example['columns'])
    for event in obs['history']:
        if event['tool']!='check':continue
        got=event['observation']
        if not got['ok'] or got['matches_example'] or got['truncated'] or got['row_count']!=example['row_count']:continue
        tree=_parse(event['sql'])
        if not isinstance(tree,exp.Select):continue
        arity=len(got['columns'])
        if arity>7 or width>arity or len(tree.expressions)!=arity:continue
        for places in permutations(range(arity),width):
            if places==tuple(range(arity)):continue
            proj=[tuple(row[j] for j in places) for row in got['rows']]
            if obs['ordered'] and proj!=list(map(tuple,rows)):continue
            if Counter(proj)!=Counter(map(tuple,rows)):continue
            # Every position must also match a public header or the expression's column.
            def fits(j,name):
                e=tree.expressions[j].unalias();wanted=name.casefold()
                return got['columns'][j].casefold()==wanted or e.name.casefold()==wanted or _sql(e).casefold()==wanted
            if all(fits(j,name) for j,name in zip(places,example['columns'])):
                yield _sql(_set(tree,expressions=[tree.expressions[j] for j in places]))
