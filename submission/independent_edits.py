"""Unbundle optional changes introduced while fixing a broken join."""
import re
from sqlglot import exp
from projection200 import _parse, _sql


def propose(obs, base):
    hist = obs['history']
    if not hist or hist[0].get('sql') != obs['issue_sql'] or hist[0]['observation'].get('ok'):
        return None
    if re.search(r'\b(distinct|unique|different|once)\b', obs['task']['question'], re.I):
        return None
    if not any(e['tool'] == 'check' and e['sql'] == base and e['observation']['matches_example'] for e in hist):
        return None
    draft, repaired = _parse(obs['issue_sql']), _parse(base)
    if any(t is None or not t.args.get('joins') for t in (draft, repaired)):
        return None
    if draft.args.get('distinct') or not repaired.args.get('distinct') or repaired.find(exp.AggFunc, exp.Subquery) or repaired.args.get('limit'):
        return None
    revised = repaired.copy()
    revised.set('distinct', None)
    # Preserve comparisons whose operand column and literal appeared together
    # in the draft. Do not transfer stale predicates into the repaired join.
    exact = {(p.this.name.lower(), p.expression.this) for p in draft.find_all(exp.EQ)
             if isinstance(p.this, exp.Column) and isinstance(p.expression, exp.Literal) and p.expression.is_string}
    for pred in revised.find_all(exp.EQ):
        a, b = pred.this, pred.expression
        if isinstance(a, exp.Lower) and isinstance(b, exp.Lower) and isinstance(a.this, exp.Column) and isinstance(b.this, exp.Literal):
            if (a.this.name.lower(), b.this.this) in exact:
                pred.replace(exp.EQ(this=a.this.copy(), expression=b.this.copy()))
    text = _sql(revised)
    return text if len(text.encode()) <= 16000 else None


class Overlay:
    def __init__(self, parent):
        self.parent = parent
        self.pair = None

    def act(self, obs):
        if self.pair is None:
            action = self.parent.act(obs)
            if action['tool'] != 'submit':
                return action
            sql = propose(obs, action['sql'])
            if sql is None:
                return action
            self.pair = (action['sql'], sql)
        base, sql = self.pair
        for e in obs['history']:
            if e['tool'] == 'check' and e['sql'] == sql:
                return {'tool': 'submit', 'sql': sql if e['observation']['matches_example'] else base}
        return {'tool': 'check', 'sql': sql} if obs['remaining'] else {'tool': 'submit', 'sql': base}
