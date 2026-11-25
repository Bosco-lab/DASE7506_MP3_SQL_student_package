"""Use declared relationships to recover the entity being counted or filtered."""
import re
from sqlglot import exp
from extra_repairs import _column, _headers, _parse, _set, _sql, _tables
from parent_agent import legacy, sql_key


def owner(column, tables, schema):
    if not isinstance(column, exp.Column):
        return None
    matches = [t for t in tables if (not column.table or t.alias_or_name.lower() == column.table.lower())
               and column.name.lower() in schema.get(t.name.lower(), {}).get('columns', {})]
    return matches[0] if len(matches) == 1 else None


def entity_edits(query, question, schema):
    tables = _tables(query)
    joins = query.args.get('joins') or []
    if len(joins) != 1 or len(tables) != 2 or joins[0].side or query.find(exp.Subquery):
        return
    on = joins[0].args.get('on')
    if not isinstance(on, exp.EQ):
        return
    for pk, fk in [(on.this, on.expression), (on.expression, on.this)]:
        parent, child = owner(pk, tables, schema), owner(fk, tables, schema)
        if parent is None or child is None or parent == child:
            continue
        edges = schema[child.name.lower()]['foreign_keys']
        if not any(e['from'].lower() == fk.name.lower() and e['to'].lower() == pk.name.lower() and
                   e['table'].lower() == parent.name.lower() for e in edges):
            continue
        group, order = query.args.get('group'), query.args.get('order')
        if group and order and query.args.get('limit') and len(group.expressions) == len(query.expressions) == 1:
            key = group.expressions[0]
            projected = query.expressions[0].unalias()
            if owner(key, tables, schema) == parent and owner(projected, tables, schema) == parent and 'name' in projected.name.lower():
                if isinstance(order.expressions[0].this, exp.Count) and re.search(r'\b(?:most|least|fewest|greatest|largest|highest)\s+(?:number|common|frequent|popular)\b', question):
                    new_order = order.copy()
                    new_order.expressions[0].set('this', exp.Count(this=exp.Star()))
                    yield 'child_frequency', _set(query, group=exp.Group(expressions=[fk]), order=new_order)
        avg = query.expressions[0].unalias() if len(query.expressions) == 1 else None
        if isinstance(avg, exp.Avg) and owner(avg.this, tables, schema) == parent and not group and not order:
            if not query.args.get('where') and re.search(r'\b(?:who|that|which) (?:have|had|has)\b', question):
                outer = exp.select(*[e.copy() for e in query.expressions]).from_(parent.copy())
                selected_keys = exp.select(fk.copy()).from_(child.copy()).subquery()
                yield 'semijoin_average', outer.where(exp.In(this=pk.copy(), query=selected_keys))


def both_edits(query, obs, schema):
    question = obs['task']['question'].casefold()
    where = query.args.get('where')
    tables = _tables(query)
    if not where or not re.search(r'\bboth\b.+\band\b', question) or query.find(exp.Subquery):
        return
    predicate = where.this
    predicates = []
    if isinstance(predicate, exp.In) and len(predicate.expressions) == 2:
        predicates = [exp.EQ(this=predicate.this.copy(), expression=e.copy()) for e in predicate.expressions]
    elif isinstance(predicate, (exp.Or, exp.And)):
        predicates = [predicate.this, predicate.expression]
    if len(predicates) != 2 or not all(isinstance(p, (exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE)) and
            isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Literal) for p in predicates):
        return
    a, b = predicates
    if a.this.name.lower() != b.this.name.lower() or a.expression.name == b.expression.name:
        return
    if any(p.expression.name.casefold() not in question for p in predicates):
        return
    for t in tables:
        bind = _headers(obs['example']['columns'], [t], schema, [])
        col = schema.get(t.name.lower(), {}).get('columns', {}).get(a.this.name.lower())
        if not bind or not col or any(e.find(exp.AggFunc) for e in bind[0]):
            continue
        sides = []
        for p in predicates:
            constraint = type(p)(this=_column(col, t.alias_or_name), expression=p.expression.copy())
            sides.append(exp.select(*[e.copy() for e in bind[0]]).from_(t.copy()).where(constraint))
        yield 'same_attribute_intersection', exp.Intersect(this=sides[0], expression=sides[1], distinct=True)


class EntityStage:
    def __init__(self, base, obs):
        self.base = base
        self.pool = []
        self.checked = {sql_key(e['sql']): e['observation']['matches_example'] for e in obs['history'] if e['tool'] == 'check'}
        schema_event = next((e for e in obs['history'] if e['tool'] == 'schema'), None)
        query = _parse(base)
        if not schema_event or not isinstance(query, exp.Select):
            return
        schema = legacy.parse_schema(schema_event['observation'])['tables']
        q = obs['task']['question'].lower()
        candidates = [*entity_edits(query, q, schema), *both_edits(query, obs, schema)]
        self.pool = [(kind, _sql(tree)) for kind, tree in candidates if sql_key(_sql(tree)) != sql_key(base)]

    def act(self, obs):
        if obs['history'] and obs['history'][-1]['tool'] == 'check':
            last = obs['history'][-1]
            self.checked[sql_key(last['sql'])] = last['observation']['matches_example']
        for kind, sql in self.pool:
            result = self.checked.get(sql_key(sql))
            if result is True:
                return {'tool': 'submit', 'sql': sql}
            if result is None and obs['remaining']:
                return {'tool': 'check', 'sql': sql}
        return {'tool': 'submit', 'sql': self.base}
