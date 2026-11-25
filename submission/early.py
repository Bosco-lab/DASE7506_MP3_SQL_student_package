"""Check strongly constrained edits while the public episode still has a budget."""
import re
from sqlglot import exp
from extra_repairs import _column, _headers, _parse, _set, _sql, _tables
from local_edits import grounded_join
from parent_agent import legacy, sql_key


def first_check(obs, info):
    if not obs['remaining'] or not info:
        return None
    history = [event for event in obs['history'] if event['tool'] == 'check']
    seen = {sql_key(event['sql']) for event in history}
    passing = [event for event in history if event['observation']['matches_example']]
    seeds = passing or history
    options = []
    for seed in reversed(seeds):
        tree = _parse(seed['sql'])
        if not isinstance(tree, exp.Select) or tree.find(exp.Subquery):
            continue
        tables = _tables(tree)
        cols = obs['example']['columns']
        projection = _headers(cols, tables, info['tables'], tree.expressions)
        if tree.args.get('group') and len(cols) < len(tree.expressions) and projection:
            options.append((6, _set(tree, expressions=projection[0])))
        for _, _, fix in grounded_join(tree, info):
            if len(_tables(fix)) > len(tables):
                options.append((9, fix))
        group = tree.args.get('group')
        if not (group and len(group.expressions) == 1 and len(tables) >= 2 and tree.args.get('limit') and tree.find(exp.Count)):
            continue
        key = group.expressions[0]
        if not isinstance(key, exp.Column) or 'name' not in key.name.lower() or len(tree.expressions) != 1:
            continue
        if not re.search(r'\b(most|least) (common|frequent|popular)\b', obs['task']['question'], re.I):
            continue
        for owner in tables:
            meta = info['tables'].get(owner.name.lower(), {})
            pks = meta.get('pks', [])
            if key.name.lower() not in meta.get('columns', {}) or len(pks) != 1:
                continue
            for child in tables:
                for fk in info['tables'].get(child.name.lower(), {}).get('foreign_keys', []):
                    if fk['table'].lower() == owner.name.lower() and fk['to'].lower() == pks[0].lower():
                        options.append((7, _set(tree, group=exp.Group(expressions=[_column(fk['from'], child.alias_or_name)]))))
    for _, candidate in sorted(options, key=lambda entry: -entry[0]):
        sql = _sql(candidate)
        if sql_key(sql) not in seen:
            return sql
    return None
