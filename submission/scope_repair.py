"""Remove a predicate that narrows a relation already grounded by the join."""
import re
from sqlglot import exp
from extra_repairs import _parse, _set, _sql
from parent_agent import sql_key


def tokens(value):
    words = re.findall(r'[a-z]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', value).lower())
    return {w.removesuffix('s') for w in words if len(w)>2} - {'the', 'has', 'have', 'and', 'with', 'for', 'from'}


def edits(query, question):
    where = query.args.get('where')
    joins = query.args.get('joins') or []
    if not where or not joins or query.find(exp.Subquery, exp.CTE):
        return
    q = tokens(question)
    join_words = set().union(*(tokens(col.name) for j in joins if j.args.get('on') for col in j.args['on'].find_all(exp.Column)))
    term = where.this
    if not isinstance(term, exp.EQ):
        return
    col, value = term.this, term.expression
    if isinstance(col, exp.Lower) and isinstance(value, exp.Lower):
        col, value = col.this, value.this
    if not isinstance(col, exp.Column):
        return
    col_words = tokens(col.name)
    if isinstance(value, exp.Boolean) and col_words and q & join_words:
        # A bool flag may share an entity word with a join key. Require a separate property not requested anywhere.
        descriptors = col_words - join_words - {'was', 'are', 'flag', 'true', 'false'}
        if descriptors and not (descriptors & q) and not (col_words & q) - join_words:
            yield 'boolean_property', _set(query, where=None)
    if isinstance(value, exp.Literal) and value.is_string:
        val_words = tokens(value.name)
        if val_words and val_words <= join_words & q and not col_words & q:
            yield 'relation_as_literal', _set(query, where=None)


class ScopeStage:
    def __init__(self, base, obs):
        self.base = base
        self.pool = []
        self.checked = {sql_key(e['sql']):e['observation']['matches_example'] for e in obs['history'] if e['tool'] == 'check'}
        tree = _parse(base)
        if isinstance(tree, exp.Select):
            self.pool = [(kind, _sql(candidate)) for kind, candidate in edits(tree, obs['task']['question'])]

    def act(self, obs):
        if obs['history'] and obs['history'][-1]['tool'] == 'check':
            last = obs['history'][-1]
            self.checked[sql_key(last['sql'])] = last['observation']['matches_example']
        for _, sql in self.pool:
            checked = self.checked.get(sql_key(sql))
            if checked is True:
                return {'tool':'submit','sql':sql}
            if checked is None and obs['remaining']:
                return {'tool':'check','sql':sql}
        return {'tool':'submit','sql':self.base}
