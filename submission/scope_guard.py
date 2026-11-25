"""Late binding of invalid or ambiguous identifiers using the purchased schema."""
import difflib
from collections import Counter
import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope
from projection200 import _words
from parent_agent import vote_key


def parse(text):
    try:
        ts = sqlglot.parse(text, read='sqlite')
        if len(ts)==1 and isinstance(ts[0],(exp.Select,exp.SetOperation)):
            return ts[0]
    except (SqlglotError,RecursionError):
        pass


def valid(tree, schema):
    if tree is None:
        return False
    try:
        qualify(tree.copy(), dialect='sqlite', schema=schema, infer_schema=False)
        return True
    except (SqlglotError,RecursionError):
        return False


def repair(tree,schema):
    if tree is None:
        return
    node=tree.copy()
    for scope in traverse_scope(node):
        owners={a.lower(): t.name.lower() for a,t in scope.sources.items() if isinstance(t,exp.Table)}
        if any(t not in schema for t in owners.values()):return
        for c in scope.columns:
            if c.find_ancestor(exp.Select) is not scope.expression:continue
            name,alias=c.name.lower(),c.table.lower()
            if alias in owners and name in schema[owners[alias]]:continue
            possible=[a for a,t in owners.items() if name in schema[t]]
            if alias and alias not in owners:
                direct=[a for a in possible if owners[a]==alias]
                if len(direct)==1:c.set('table',exp.to_identifier(direct[0],quoted=True))
                # A qualifier not in the current scope could be a correlation.
            elif len(possible)==1:
                c.set('table',exp.to_identifier(possible[0],quoted=True))
            elif not possible:
                # Only a unique close spelling in a known relation.
                hits=sorted([(difflib.SequenceMatcher(None,name,k).ratio(),a,k) for a,t in owners.items() if not alias or a==alias for k in schema[t]],reverse=True)
                if hits and hits[0][0]>=.82 and (len(hits)==1 or hits[0][0]-hits[1][0]>.10):
                    _,a,k=hits[0];c.set('table',exp.to_identifier(a,quoted=True));c.set('this',exp.to_identifier(k,quoted=True))
    return node


def options(obs, incumbent, force=False):
    raw=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),[])
    if not raw:return []
    checked={vote_key(e['sql']):e['observation'] for e in obs['history'] if e['tool']=='check'}
    if checked.get(vote_key(incumbent),{}).get('ok') and not force:return []
    schema={t['name'].lower():{c['name'].lower():'UNKNOWN' for c in t['columns']} for t in raw}
    base=parse(incumbent)
    if valid(base,schema) and not force:return []
    hints=next((e['observation'] for e in obs['history'] if e['tool']=='repair_hint'),[])
    seeds=list(dict.fromkeys([incumbent,*hints,obs['issue_sql'],*[e['sql'] for e in reversed(obs['history']) if e['tool']=='check']]))
    hdr=[h.lower() for h in obs['example']['columns']];pool={}
    def add(node,score):
        if not valid(node,schema):return
        sql=node.sql(dialect='sqlite',comments=False)
        k=vote_key(sql)
        if len(sql.encode())>16000 or k in checked and not checked[k]['matches_example'] or k==vote_key(incumbent):return
        if len(node.selects)!=len(hdr):return
        pool[sql]=max(score,pool.get(sql,-100))
    for i,seed in enumerate(seeds):
        tree=parse(seed)
        if tree is None:continue
        n=repair(tree,schema)
        if n is None:continue
        rank=5*(i==0)+2*(seed in hints)-i*.1
        add(n,rank)
        # Bind output slots by their exact observed labels. Fix an invalid
        # projection without importing its stale qualifier into the new query.
        if not isinstance(n,exp.Select) or not n.args.get('from'):continue
        tabs=[n.args['from'].this,*(j.this for j in n.args.get('joins',[]))]
        if not all(isinstance(t,exp.Table) for t in tabs):continue
        choices=[[(t.alias_or_name,h) for t in tabs if h in schema.get(t.name.lower(),{})] for h in hdr]
        if choices and all(len(c)==1 for c in choices):
            edit=n.copy();edit.set('expressions',[exp.column(c,table=t,quoted=True) for [(t,c)] in choices]);add(edit,rank-.5)
    return sorted(pool,key=pool.get,reverse=True)


class Overlay:
    def __init__(self,parent):
        self.parent=parent
        self.pool=None

    def act(self,obs):
        if self.pool is None:
            action=self.parent.act(obs)
            if action['tool']!='submit':return action
            self.base=action['sql'];self.pool=options(obs,self.base)[:3]
        for sql in self.pool:
            key=vote_key(sql)
            prev=next((e for e in obs['history'] if e['tool']=='check' and vote_key(e['sql'])==key),None)
            if prev:
                if prev['observation']['matches_example']:return {'tool':'submit','sql':prev['sql']}
            else:return {'tool':'check' if obs['remaining'] else 'submit','sql':sql}
        return {'tool':'submit','sql':self.base}
