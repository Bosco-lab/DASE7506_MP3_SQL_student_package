"""Schema-bound predicates and typed literals, inferred from the public episode."""
import re
from sqlglot import exp
from values_schema import Catalog, affinity, lit, qi, ref, words
from witness import parse


def options(obs):
    from ordered200 import ranked_rows
    yield from ranked_rows(obs)
    q=obs['task']['question'].lower()
    cat=Catalog(obs)
    checks=[e for e in obs['history'] if e['tool']=='check']
    hints=next((e['observation'] for e in obs['history'] if e['tool']=='repair_hint'),[])
    texts=list(dict.fromkeys([*hints,*[e['sql'] for e in reversed(checks)]]))
    is_count=len(cat.headers)==1 and cat.headers[0].lower().replace(' ','')=='count(*)'
    def count(t):return 'SELECT COUNT(*) FROM '+qi(t)
    if is_count:
        cmp=re.search(r'\b(older|younger|taller|shorter) than (?:all|every) ([a-z]+)',q)
        if cmp:
            metric='age' if cmp[1] in ('older','younger') else 'height'
            shorter=cmp[1] in ('younger','shorter')
            for t,cols in cat.cols.items():
                m=cat.owner(t,metric); roles=[c for c in cols if set(words(c))&{'job','occupation','profession','role'}]
                if m and len(roles)==1:
                    agg,op=('MIN','<') if shorter else ('MAX','>')
                    yield 'role_extremum',f'{count(t)} WHERE {qi(m[1])} {op} (SELECT {agg}({qi(m[1])}) FROM {qi(t)} WHERE LOWER({qi(roles[0])}) = {lit(cmp[2].rstrip("s"))})'
        type_q=re.search(r'\bhow many (.+?) (?:types|categories) (?:are there|do we have)',q)
        if type_q:
            types=set(words(type_q[1])); candidates=[t for t in cat.cols if types<=set(words(t)) and set(words(t))&{'ref','type','category'}]
            if len(candidates)==1:yield 'reference_types',count(candidates[0])
    unit=re.search(r'\b(\d+)\s*[- ]?\s*(minutes?|hours?)\b',q)
    quoted=re.findall(r'''["']([^"']+)["']''',obs['task']['question'])
    for text in texts:
        tree=parse(text)
        if not isinstance(tree,exp.Select):continue
        where=tree.args.get('where')
        if not where:continue
        pred=where.this
        if isinstance(pred,exp.EQ) and isinstance(pred.this,exp.Column) and isinstance(pred.expression,exp.Literal):
            term=pred.expression.name; col=pred.this
            for t,c in cat.resolve(col,tree):
                typ=cat.cols[t][c]['type']
                if unit and unit[1]==term and affinity(typ)=='text' and set(words(c))&{'duration','time','length','runtime'}:
                    fixed=tree.copy();fixed.set('where',exp.Where(this=exp.Like(this=col.copy(),expression=exp.Literal.string(term+':%'))));yield 'clock_prefix',fixed.sql(dialect='sqlite')
                width=re.search(r'\((\d+)\)',typ)
                if is_count and width and len(term)>int(width[1]) and re.fullmatch(r'[A-Za-z]+-\d+',term) and term.lower() in q:
                    related=[]
                    for fk,u,key,_ in cat.graph[t]:
                        if fk.lower()!=c.lower():continue
                        for name,meta in cat.cols[u].items():
                            w=re.search(r'\((\d+)\)',meta['type'])
                            if name.lower()!=key.lower() and 'code' in words(name) and w and int(w[1])>=len(term):related.append((u,name))
                    if len(set(related))==1:
                        f=related[0]; b=cat.select([f],anchor=t,count=True)
                        if b:yield 'foreign_code_width',b[0]+' WHERE '+ref(*f)+' = '+lit(term)
                if quoted and term in quoted and len(quoted)==1 and len(term)>2 and re.match(r'^[\w ,.-]+$',term):
                    phrase=q.split(term.lower())[0].rstrip(" '\""); terms=words(phrase)[-5:]
                    for u,cols in cat.cols.items():
                        for k in cols:
                            kw=words(k)
                            if len(kw)<2 or len(kw)>5 or kw!=terms[-len(kw):] or (u,k)==(t,c):continue
                            if set(kw)<=set(words(c)):continue
                            b=cat.select([(u,k)],count=is_count,distinct=bool(tree.args.get('distinct')))
                            if b and not tree.args.get('order') and not tree.args.get('group'):
                                yield 'explicit_literal_column',b[0]+' WHERE '+ref(u,k)+' = '+lit(term)
        if is_count and isinstance(pred,exp.In) and isinstance(pred.this,exp.Column) and all(isinstance(x,exp.Literal) for x in pred.expressions) and ' or ' in q:
            for t,c in cat.resolve(pred.this,tree):
                if 'degree' not in words(c):continue
                names=[x.name for x in pred.expressions];out=[]
                for n in names:
                    options={'master':['MA','MS','M.A.','M.S.'],'bachelor':['BA','BS','B.A.','B.S.']}.get(n.lower().rstrip('s'),[])
                    out.extend([n,*options])
                if len(out)>len(names):yield 'degree_alias',count(t)+' WHERE LOWER('+qi(c)+') IN ('+', '.join(lit(v.lower()) for v in out)+')'
        # Use a uniquely named table for the answer entity when multiple tables share the header.
        if isinstance(pred,exp.EQ) and pred.expression and isinstance(pred.expression,exp.Literal) and len(cat.headers)<=3 and not any(tree.args.get(k) for k in ('order','limit','group')):
            term=pred.expression.name; fields=cat.resolve(pred.this,tree)
            if term.lower() not in q or len(fields)!=1:continue
            roots=[t for t in cat.cols if all(cat.owner(t,k) for k in cat.headers)]; root_match=[t for t in roots if len(words(t))>1 and set(words(t))<=cat.qw]
            if len(roots)>1 and len(root_match)==1 and fields[0][0]!=root_match[0]:
                b=cat.select(fields,anchor=root_match[0],distinct=bool(tree.args.get('distinct')))
                if b:yield 'named_owner_join',b[0]+' WHERE '+ref(*fields[0])+' = '+lit(term)
    excl=re.search(r'\bwithout (?:a|an) (\w+) (\w+)\b',q)
    if excl and not is_count:
        kind,noun=excl.groups();fields=[(t,c) for t,cols in cat.cols.items() if words(noun)[0] in words(t) for c in cols if {'type','code'}<=set(words(c)) and affinity(cols[c]['type'])=='text']
        if len(fields)==1:
            root=cat.select([]); joined=cat.select(fields,anchor=root[1]) if root else None
            if joined and fields[0][0]!=root[1]:yield 'category_absence',root[0]+' EXCEPT '+joined[0]+' WHERE LOWER('+ref(*fields[0])+') = '+lit(kind)
