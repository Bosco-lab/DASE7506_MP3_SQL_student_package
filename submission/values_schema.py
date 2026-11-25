"""Resolve answer columns and build minimal join paths from the observed schema."""
import re
from collections import deque
from sqlglot import exp


def qi(s):
    return '"' + s.replace('"', '""') + '"'


def lit(s):
    return "'" + str(s).replace("'", "''") + "'"


def ref(t, c):
    return qi(t) + '.' + qi(c)


def words(s):
    s = re.sub('([a-z])([A-Z])', r'\1 \2', s)
    return [w[:-3]+'y' if w.endswith('ies') else w[:-1] if len(w)>3 and w.endswith('s') and not w.endswith('ss') else w for w in re.findall(r'[a-z0-9]+', s.lower())]


def affinity(declared):
    s = declared.upper()
    if any(x in s for x in ('TEXT','CHAR','CLOB')): return 'text'
    if 'BOOL' in s: return 'bool'
    if any(x in s for x in ('INT','REAL','NUM','DEC','FLOAT','DOUBLE')): return 'number'
    return 'other'


class Catalog:
    def __init__(self, obs):
        schema = next(e['observation'] for e in obs['history'] if e['tool']=='schema')
        self.cols = {t['name']:{c['name']:c for c in t['columns']} for t in schema}
        self.graph = {t:[] for t in self.cols}
        def edge(t,c,u,k,weight):
            if t==u or (t,c) not in self.fields or (u,k) not in self.fields: return
            for a,b,d,e in ((t,c,u,k),(u,k,t,c)):
                if not any(x[:3]==(b,d,e) for x in self.graph[a]): self.graph[a].append((b,d,e,weight))
        self.fields = {(t,c) for t,cols in self.cols.items() for c in cols}
        for t in schema:
            for f in t['foreign_keys']:
                target=self.owner(f['table'], f['to'] or '')
                if target: edge(t['name'],f['from'],*target,0)
        for t,cols in self.cols.items():
            for c,meta in cols.items():
                if not meta.get('pk') or c.lower()=='id': continue
                for u in self.cols:
                    target=self.owner(u,c)
                    if target: edge(t,c,*target,1)
        self.q = obs['task']['question']; self.qw=set(words(self.q)); self.headers=obs['example']['columns']

    def owner(self, t, c):
        return next(((a,b) for a,cols in self.cols.items() if a.lower()==t.lower() for b in cols if b.lower()==c.lower()),None)

    def path(self, root, dest):
        if root==dest: return []
        queue=deque([(root,[],{root})]); paths=[]
        while queue:
            t,path,seen=queue.popleft()
            if len(path)>=3: continue
            for c,u,k,inferred in self.graph[t]:
                if u in seen: continue
                more=path+[(t,c,u,k,inferred)]
                if u==dest: paths.append(more)
                else: queue.append((u,more,seen|{u}))
        if paths: return min(paths,key=lambda p:len(p)+0.6*sum(e[4] for e in p))

    def select(self, fields, anchor=None, *, count=False, distinct=False):
        # Pick a root by exact public header names, question ownership and FK distance.
        roots=list(self.cols) if anchor is None else [anchor]
        options=[]
        for root in roots:
            if not count and not all(self.owner(root,h) for h in self.headers): continue
            needed=list(dict.fromkeys(t for t,c in fields)); joins=[]; used={root}
            for dest in needed:
                if dest in used: continue
                path=self.path(root,dest)
                if path is None: break
                for t,c,u,k,_ in path:
                    if u in used: continue
                    joins.append(' JOIN '+qi(u)+' ON '+ref(t,c)+' = '+ref(u,k));used.add(u)
            else:
                proj='COUNT(*)' if count else ', '.join(ref(*self.owner(root,h)) for h in self.headers)
                sql='SELECT '+('DISTINCT ' if distinct and not count else '')+proj+' FROM '+qi(root)+''.join(joins)
                match=len(set(words(root)) & self.qw)
                options.append((match-1.6*len(joins),sql,root))
        if options: return max(options,key=lambda v:v[0])[1:]

    def resolve(self, col, tree):
        if not isinstance(col,exp.Column): return []
        tables=[t for t in tree.find_all(exp.Table) if not col.table or t.alias_or_name.lower()==col.table.lower()]
        local=[self.owner(t.name,col.name) for t in tables];local=list(dict.fromkeys(x for x in local if x))
        return local or [(a,b) for a,cols in self.cols.items() for b in cols if b.lower()==col.name.lower()]
