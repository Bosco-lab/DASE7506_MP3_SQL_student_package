"""Plan one limited public lookup. Rank bindings before looking at any returned cell."""
from dataclasses import dataclass
import re
from sqlglot import exp
from values_schema import affinity, lit, qi, ref, words


@dataclass
class Lookup:
    kind: str
    fields: list
    terms: list
    base: object
    extra: object = None

    def query(self):
        # No unbounded dump. At most 12 schema candidates, 4 matching distinct
        # cells per field, eight short cells across the entire public observation.
        parts=[]
        for i,(t,c) in enumerate(self.fields[:12]):
            col=qi(c); val=f'LOWER(CAST({col} AS TEXT))'; vocab=set()
            for term in self.terms: vocab.update(variants(term))
            if self.kind=='boolean_code': vocab.update(POS|NEG)
            if self.kind=='label_number':vocab.update((' '.join(words(c))+' '+self.terms[0],))
            cond=val+' IN ('+', '.join(lit(v) for v in sorted(vocab))+')'
            parts.append('SELECT * FROM (SELECT DISTINCT '+str(i)+' AS k, '+col+' AS v FROM '+qi(t)+' WHERE '+cond+' AND LENGTH('+col+') <= 80 ORDER BY v LIMIT 4)')
        return 'SELECT k,v FROM ('+' UNION ALL '.join(parts)+') ORDER BY k,v LIMIT 8'

    def finish(self, rows):
        hits=[(self.fields[k],str(v)) for k,v in rows if isinstance(k,int) and k<len(self.fields)]
        def best(term,fields=None):
            opts=[(f,v) for f,v in hits if v.lower() in variants(term) and (fields is None or f in fields)]
            return opts[0] if opts else None
        test=''; anti=False
        if self.kind=='route_pair':
            pair=[best(v,fields) for v,fields in zip(self.terms,self.extra)]
            if not all(pair):return
            fields=[f for f,v in pair];test=' AND '.join(ref(*f)+' = '+lit(v) for f,v in pair)
        elif self.kind=='quantified_category':
            pair=best(self.terms[0]);
            if not pair:
                roles=[f for f in self.fields if {'job','occupation','profession','role'} & set(words(f[1]))]
                if len(roles)!=1 or not re.search(r'(er|or|ist|ian)$',self.terms[0]):return
                pair=(roles[0],self.terms[0])
            (t,c),v=pair; num,op=self.extra
            fields=[(t,c)]
            agg='MAX' if op=='>' else 'MIN'
            test=f'{ref(t,num)} {op} (SELECT {agg}({qi(num)}) FROM {qi(t)} WHERE {qi(c)} = {lit(v)})'
        elif self.kind=='boolean_code':
            truth=self.extra; vals=[(f,v) for f,v in hits if v.lower() in (POS if truth else NEG)]
            if not vals:return
            f,v=vals[0];fields=[f];test=ref(*f)+' = '+lit(v)
        elif self.kind=='label_number':
            v=self.terms[0]; pairs=[(f,x) for f,x in hits if x.lower()==' '.join(words(f[1]))+' '+v];
            if not pairs or any(x==v for _,x in hits):return
            f,x=pairs[0];fields=[f];test=ref(*f)+self.extra+lit(x)
        elif self.kind=='category_list':
            f=self.fields[0]; domain={v.lower():v for ff,v in hits if ff==f};vs=[]
            for term in self.terms:
                if term.lower() in domain:vs.append(domain[term.lower()]);continue
                alt=sorted(domain.keys() & variants(term))
                if len(alt)==1:vs.append(domain[alt[0]])
                elif not alt and self.extra:vs.append(term)
                else:return
            if [v.lower() for v in vs]==[v.lower() for v in self.terms] and not self.extra:return
            fields=[f];test=ref(*f)+' IN ('+', '.join(map(lit,vs))+')'
        elif self.kind=='morphology':
            f=self.fields[0];term=self.terms[0]; pair=best(term)
            if pair: f,v=pair
            else:
                # Noun to past-participle for result/status enums. Public DB may
                # contain no instances of a requested result; still check the edit.
                if not term.lower().endswith('ation') or not {'result','status','outcome'} & set(words(f[1])):return
                v=term[:-3]+'ed'
            if v.lower()==term.lower():return
            fields=[f];test='LOWER('+ref(*f)+') = '+lit(v.lower())
        else:
            term=self.terms[0];p=best(term)
            if not p and self.kind=='typed_code' and len(self.fields)==1:p=(self.fields[0],term)
            if not p:return
            f,v=p;fields=[f];test=ref(*f)+' = '+lit(v)
            if self.kind=='typed_code' and f in self.extra:return
            anti=self.kind=='excluded_category'
        b=self.base(fields)
        if b:
            sql,root=b
            if anti:
                start=self.base([])
                if not start or start[1]!=root:return
                sql=start[0]+' EXCEPT '+sql
            yield self.kind,sql+' WHERE '+test


POS={'1','y','yes','true','t','on'}
NEG={'0','n','no','false','f','off'}

def variants(term):
    s=term.lower(); vs={s}
    if s.endswith('ation'):vs.add(s[:-3]+'ed')
    if s.endswith('ies'):vs.add(s[:-3]+'y')
    if s.endswith('ian'):vs.add(s[:-1]);vs.add(s[:-3])
    if s.endswith('s') and len(s)>4:vs.add(s[:-1])
    if s.rstrip('s') in ('master','bachelor'):vs.update((s[0]+'a',s[0]+'s',s[0]+'.a.',s[0]+'.s.'))
    return vs


def plan(cat,trees,passing,obs):
    q=cat.q; low=q.lower();count=bool(re.fullmatch(r'count\(\*\)',cat.headers[0],re.I) and len(cat.headers)==1)
    distinct='distinct' in cat.qw or 'unique' in cat.qw
    def base(fields,anchor=None):return cat.select(fields,anchor,count=count,distinct=distinct)
    facts=[]
    for tree in trees:
        wh=tree.args.get('where')
        if not wh: continue
        for e in wh.find_all(exp.EQ,exp.NEQ,exp.Like,exp.In):
            vals=e.expressions if isinstance(e,exp.In) else [e.expression]
            if not vals or not all(isinstance(v,exp.Literal) for v in vals):continue
            for f in cat.resolve(e.this,tree):
                rec=(f,type(e),tuple(v.name.strip('%') for v in vals))
                if rec not in facts:facts.append(rec)
    quant=re.search(r'\b(older|younger|taller|shorter) than every ([\w-]+)',low)
    if count and quant and not passing:
        measure='age' if quant[1] in ('older','younger') else 'height';op='>' if quant[1] in ('older','taller') else '<'
        for t,cols in cat.cols.items():
            m=cat.owner(t,measure);fs=[(t,c) for c in cols if affinity(cols[c]['type'])=='text']
            if m and fs:return Lookup('quantified_category',fs,[quant[2]],lambda fields:base(fields,t),(m[1],op))
    for f,op,vs in facts:
        t,c=f; typ=affinity(cat.cols[t][c]['type']); v=vs[0].lower(); width=re.search(r'\((\d+)\)',cat.cols[t][c]['type'])
        if count and not passing and (v in {'yes','no','true','false'} or typ=='bool' or re.search(r'\b(logged in|signed in|active|enabled|disabled)\b',low)) and not (' or ' in low or ' and ' in low):
            truth=not bool(re.search(r"\b(no|not|never|without|disabled)\b|n't\b",low))
            return Lookup('boolean_code',[f],[v],lambda fields:base(fields,t),truth)
        if typ=='text' and v.isdigit() and len(v)<=3 and op in (exp.NEQ,exp.EQ) and set(words(c)) & cat.qw and not passing:
            return Lookup('label_number',[f],[v],base,' <> ' if op==exp.NEQ else ' = ')
        if op==exp.In and len(vs)<=4 and all(v.lower() in low for v in vs) and (' or ' in low):
            needs_alias=any(len(variants(v))>2 for v in vs); local=all(cat.owner(t,h) for h in cat.headers) and not ({'each','per','order','sort'} & cat.qw) and not any(set(words(x.name)) & cat.qw for x in trees[0].find_all(exp.Table) if x.name.lower()!=t.lower())
            if needs_alias or (local and any(tree.args.get('joins') for tree in trees)):
                return Lookup('category_list',[f],list(vs),lambda fields:base(fields,t),local)
        if len(vs)==1 and v in low and v.endswith('ation') and typ=='text' and {'status','outcome','result'} & set(words(c)):
            return Lookup('morphology',[f],[vs[0]],base)
        if width and typ=='text' and len(v)>int(width[1]) and re.search(r'[a-z]+[- ]\d+',v) and v in low and count:
            fs=[]
            for _,u,_,_ in cat.graph[t]:
                if not set(words(u)) & cat.qw:continue
                for d,meta in cat.cols[u].items():
                    w=re.search(r'\((\d+)\)',meta['type'])
                    if affinity(meta['type'])=='text' and 'code' in words(d) and (not w or int(w[1])>=len(v)):fs.append((u,d))
            fs=list(dict.fromkeys(fs))
            if fs:return Lookup('typed_code',fs,[vs[0]],lambda fields:base(fields,t),[f])
    route=re.search(r'\bfrom ([\w -]+?) to ([\w -]+?)[.?!]?$',q,re.I)
    if route and not passing:
        for t,cols in cat.cols.items():
            if not all(cat.owner(t,h) for h in cat.headers):continue
            fs=[[(t,c) for c in cols if set(words(c))&syn] for syn in ({'origin','source','from','start'},{'destination','to','end','target'})]
            if all(fs):return Lookup('route_pair',fs[0]+fs[1],list(route.groups()),lambda fields:base(fields,t),fs)
    ex=re.search(r'\bwithout (?:a|an) (\w+) (\w+)\b',low)
    if ex:
        fs=[(t,c) for t,cols in cat.cols.items() if words(ex[2])[0] in words(t) for c,meta in cols.items() if 'type' in words(c) and affinity(meta['type'])=='text']
        if fs:return Lookup('excluded_category',fs,[ex[1]],base)
    if passing or count or not facts or obs['ordered'] or {'most','least','order','ordered','sort','each','per','without','between','both','not','or'} & cat.qw:return
    # A single categorical literal and the entire projection live on a native
    # relation. Current failed checks may have joined an unrelated fact table.
    terms={vs[0] for _,op,vs in facts if op==exp.EQ and len(vs)==1 and vs[0].lower() in low and not vs[0].isdigit()}
    if len(terms)==1 and any(tree.args.get('joins') for tree in trees):
        v=next(iter(terms));fs=[f for f,op,vs in facts if vs==(v,) and all(cat.owner(f[0],h) for h in cat.headers)]
        if fs:return Lookup('local_category',list(dict.fromkeys(fs)),[v],base)
