"""Separate row-source changes from optional duplicate elimination."""
import re
from sqlglot import exp
from projection200 import _parse, _sql, _words


def proposal(obs, base):
    hist = obs['history']
    if not hist or hist[0].get('sql') != obs['issue_sql']:
        return
    checks = {e['sql']: e['observation'] for e in hist if e['tool'] == 'check'}
    result = checks.get(base)
    if not result or not result['matches_example']:
        return
    t = _parse(base)
    if t is None or t.find(exp.AggFunc, exp.Subquery) or t.args.get('group') or t.args.get('limit'):
        return
    draft = _parse(obs['issue_sql'])
    if draft is None:
        return
    before = checks.get(obs['issue_sql'], {})
    if t.args.get('distinct') and not draft.args.get('distinct') and not t.args.get('joins') and draft.args.get('joins'):
        if before.get('ok') and not before['matches_example'] and before['row_count'] < obs['example']['row_count']:
            # DISTINCT cannot add the missing rows. The change in relation explains the gain.
            if re.search(r'\b(distinct|unique|different|once)\b', obs['task']['question'], re.I):
                return
            fixed = t.copy();fixed.set('distinct', None)
            return _sql(fixed)
    if t.args.get('distinct') or t.args.get('joins') or len(t.expressions) != 1 or not isinstance(t.expressions[0].unalias(), exp.Column):
        return
    col = t.expressions[0].unalias()
    wh = t.args.get('where')
    if not wh or not isinstance(wh.this, exp.Like) or not isinstance(wh.this.this, exp.Column) or wh.this.this.name.lower() != col.name.lower():
        return
    # An entity-valued attribute is both the answer and the search domain.
    # Require a real passing rival that only changes uniqueness, with agreement
    # among the hints. A name/label of the table itself is not a foreign entity.
    tbl = t.find(exp.Table)
    if tbl is None:
        return
    entity = set(_words(col.name)) - set(_words(tbl.name)) - {'name','title','id','code','description','label'}
    if not entity & set(_words(obs['task']['question'])):
        return
    hints = next((e['observation'] for e in hist if e['tool']=='repair_hint'), [])
    if not hints or len(set(hints)) != 1:
        return
    h = _parse(hints[0]); old=checks.get(hints[0])
    if h is not None and h.args.get('distinct') and old and old['matches_example']:
        h.set('distinct',None)
        if h == t:
            return hints[0]


class Overlay:
    def __init__(self,parent):
        self.parent=parent
        self.pending=None

    def act(self,obs):
        if self.pending is None:
            a=self.parent.act(obs)
            if a['tool']!='submit':return a
            sql=proposal(obs,a['sql'])
            if not sql or len(sql.encode())>16000:return a
            self.pending=(sql,a['sql'])
        sql,base=self.pending
        for e in obs['history']:
            if e['tool']=='check' and e['sql']==sql:
                return {'tool':'submit','sql':sql if e['observation']['matches_example'] else base}
        if obs['remaining']:return {'tool':'check','sql':sql}
        return {'tool':'submit','sql':base}
