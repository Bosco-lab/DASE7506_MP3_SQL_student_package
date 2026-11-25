"""Compile comparison quantifiers and negative entity conditions into SQLite."""
import re
from sqlglot import exp
from extra_repairs import _column, _headers, _parse, _set, _sql, _tables
from parent_agent import legacy, sql_key


def comparison(tree, question):
    for obj in tree.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE):
        target = obj.expression
        if not isinstance(target, (exp.All,exp.Any)):
            continue
        sub = target.find(exp.Select)
        if not sub or len(sub.expressions)!=1:
            continue
        col = sub.expressions[0].unalias()
        if not isinstance(col,exp.Column):
            continue
        greater = isinstance(obj,(exp.GT,exp.GTE))
        universal = isinstance(target,exp.All) or bool(re.search(r'\b(before|earlier than|after|later than) (?:the date of )?any\b',question))
        agg = exp.Max if greater == universal else exp.Min
        inner = _set(sub, expressions=[agg(this=col.copy())])
        new = tree.copy()
        match = next(x for x in new.find_all(type(obj)) if _sql(x)==_sql(obj))
        match.set('expression',inner.subquery())
        yield 'quantified_comparison',new


def stem(s):
    return {w.rstrip('s')[:5] for w in re.findall('[a-z]+',s.lower()) if len(w)>2}


def subtract(tree, obs, schema):
    q=obs['task']['question'].lower()
    if not re.search(r"\b(no|without|never|not|don't|doesn't)\b",q):
        return
    tables=_tables(tree)
    if not tables: return
    projections=_headers(obs['example']['columns'],tables[:1],schema,tree.expressions)
    if not projections or any(e.find(exp.AggFunc) for e in projections[0]):return
    columns=projections[0];base=tables[0]
    lhs=exp.select(*[c.copy() for c in columns]).from_(base.copy())
    where=tree.args.get('where');joins=tree.args.get('joins') or []
    if where and isinstance(where.this,exp.Is) and isinstance(where.this.expression,exp.Null) and len(joins)==1 and joins[0].side=='LEFT':
        inner=joins[0].copy();inner.set('side',None)
        rhs=lhs.copy();rhs.set('joins',[inner]);yield 'relation_difference',exp.Except(this=lhs.copy(),expression=rhs,distinct=True)
    if where and joins and tree.args.get('having') and isinstance(where.this,exp.Not) and isinstance(where.this.this,exp.Like):
        cmp=where.this.this
        v=cmp.expression
        if isinstance(v,exp.Literal) and v.is_string and v.name.strip('%').lower() in q:
            joins=[j.copy() for j in joins]
            for j in joins:j.set('side',None)
            rhs=_set(lhs,joins=joins,where=exp.Where(this=exp.EQ(this=cmp.this.copy(),expression=exp.Literal.string(v.name.strip('%')))))
            yield 'negative_filter',exp.Except(this=lhs.copy(),expression=rhs,distinct=True)
    if len(tables)!=1: return
    # A same-table value group is excluded if ANY row with the projected value satisfies the disqualifier.
    if re.search(r'\b(no|without)\b',q):
        threshold=re.search(r'\b(over|above|greater than|more than|under|below|less than)\s+(\d+)\b',q)
        if threshold:
            excluded=stem(' '.join(obs['example']['columns'])); nouns=stem(q)
            cols=[v for c,v in schema[base.name.lower()]['columns'].items() if stem(c)&(nouns-excluded)-{'id','name'}]
            if len(cols)==1:
                op=exp.LT if threshold[1] in ('under','below','less than') else exp.GT
                rhs=lhs.copy().where(op(this=_column(cols[0],base.alias_or_name),expression=exp.Literal.number(threshold[2])))
                yield 'range_difference',exp.Except(this=lhs.copy(),expression=rhs,distinct=True)
    if not where or not tree.find(exp.In,exp.Exists):return
    pks=schema.get(base.name.lower(),{}).get('pks',[])
    if len(pks)!=1:return
    others=[(name,meta) for name,meta in schema.items() if name!=base.name.lower() and stem(name)&stem(q)]
    candidates=[]
    for name,meta in others:
        for edge in meta['foreign_keys']:
            if edge['table'].casefold()!=base.name.casefold() or edge['to'].casefold()!=pks[0].casefold():continue
            source=exp.to_table(meta['name'],quoted=True)
            left=_column(pks[0],base.alias_or_name);right=_column(edge['from'],source.alias_or_name)
            rhs=lhs.copy().join(source,on=exp.EQ(this=left,expression=right))
            candidates.append((bool(stem(edge['from'])&stem(q)),exp.Except(this=lhs.copy(),expression=rhs,distinct=True)))
    if len(candidates)==1 or sum(v for v,_ in candidates)==1:
        yield 'negative_relation',max(candidates,key=lambda e:e[0])[1]


def hypotheses(obs):
    raw=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
    if not raw:return []
    schema=legacy.parse_schema(raw)['tables']; q=obs['task']['question'].lower()
    checks=[e for e in obs['history'] if e['tool']=='check']; hints=next((e['observation'] for e in obs['history'] if e['tool']=='repair_hint'),[])
    checked={sql_key(e['sql']):e['observation']['matches_example'] for e in checks};out={}
    for seed in dict.fromkeys([*hints,obs['issue_sql']]):
        tree=_parse(seed)
        if not isinstance(tree,exp.Select):continue
        for kind,c in [*comparison(tree,q),*subtract(tree,obs,schema)]:
            s=_sql(c);key=sql_key(s)
            if checked.get(key) is not False:out.setdefault(key,(kind,s))
    return list(out.values())[:4]
