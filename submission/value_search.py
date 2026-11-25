"""Resolve a suspected misplaced literal using one public value lookup and schema FKs."""
import re
from sqlglot import exp
from extra_repairs import _parse, _sql, _tables, _headers, _set, _column


def quote(s):
    return '"' + s.replace('"','""') + '"'


def literal(s):
    return "'" + s.replace("'", "''") + "'"


def predicates(tree):
    where=tree.args.get('where')
    if where:
        for item in where.find_all(exp.EQ,exp.NEQ):
            if isinstance(item.this,exp.Column) and isinstance(item.expression,exp.Literal) and item.expression.is_string:
                yield item


def lookup(obs,info):
    schema=info['tables'];graph=info['fk_graph'];checks=[e for e in obs['history'] if e['tool']=='check'];q=obs['task']['question'].lower()
    passed=[e for e in checks if e['observation']['matches_example']]
    if passed and obs['example']['row_count']>0 and obs['example']['rows']!=[[0]]: return
    seeds=passed or [e for e in checks if e['sql']!=obs['issue_sql']]+checks[:1]
    details=[]; fields={}
    for seed in seeds:
        tree=_parse(seed['sql'])
        if not isinstance(tree,exp.Select) or tree.find(exp.Subquery):continue
        sources=_tables(tree); names={t.name.casefold() for t in sources}
        for pred in predicates(tree):
            text=pred.expression.name
            if len(text)<2 or len(text)>48 or not re.search(r'[A-Za-z]',text):continue
            details.append((tree,pred,text)); known=pred.this.name.casefold(); related=names|{fk['to_table'].casefold() for n in names for fk in graph.get(n,[])}
            for t in sorted(related):
                if t not in schema:continue
                for c,real in schema[t]['columns'].items():
                    if re.search(r'(^|_)id($|_)|date|time|price|quantity',c) and c!=known:continue
                    score=4*(c==known)+2*(t in names)+sum(w in q for w in re.findall('[a-z]+',c) if len(w)>2)
                    name=(schema[t]['name'],real,text)
                    fields[name]=max(fields.get(name,-1),score)
    if not details:return
    fields=sorted(fields,key=lambda x:-fields[x])[:18]
    sub=[]
    for i,(t,c,text) in enumerate(fields):
        col=quote(c); tab=quote(t); match=literal(text.casefold()); stem=literal(text[:max(4,len(text)-3)].casefold())
        # Exact > substring > common stem. At most the eight best hits reach the agent.
        s=f'SELECT {i} AS k, {col} AS v, CASE WHEN lower({col})={match} THEN 3 WHEN instr(lower({col}),{match})>0 THEN 2 ELSE 1 END AS strength FROM {tab} WHERE typeof({col})=\'text\' AND instr(lower({col}),{stem})>0 GROUP BY {col}'
        if sum(map(len,sub))+len(s)+30*len(sub)>15400:break
        sub.append(s)
    if not sub:return
    sql='SELECT k,v,strength FROM ('+' UNION ALL '.join(sub)+') ORDER BY strength DESC,k LIMIT 8'
    return sql,fields,details


def repaired(obs,info,plan,rows):
    _,fields,details=plan; schema=info['tables'];graph=info['fk_graph'];out={};q=obs['task']['question'].lower()
    for k,value,strength in rows:
        t,c,text=fields[k]; target=t.casefold()
        for i,(tree,pred,old) in enumerate(details):
            if text!=old:continue
            sources=_tables(tree); by_name={s.name.casefold():s for s in sources}; current=pred.this.name.casefold()
            if target in by_name:
                ref=by_name[target];joins=tree.args.get('joins') or []
            else:
                paths=[(s,edge) for s in sources for edge in graph.get(s.name.casefold(),[]) if edge['to_table'].casefold()==target]
                if not paths:continue
                src,edge=paths[0]; ref=exp.to_table(t,quoted=True);on=exp.EQ(this=_column(edge['from_col'],src.alias_or_name),expression=_column(edge['to_col'],t));joins=[*(tree.args.get('joins') or []),exp.Join(this=ref,on=on)]
            if c.lower()==current and old==value and len(joins)==len(tree.args.get('joins') or []):continue
            node=type(pred)(this=_column(c,ref.alias_or_name),expression=exp.Literal.string(value)); new=_set(tree,joins=joins)
            for p in predicates(new):
                if _sql(p)==_sql(pred):p.replace(node);break
            choices=_headers(obs['example']['columns'],_tables(new),schema,new.expressions)
            for j,n in enumerate([*[_set(new,expressions=p) for p in choices[:1]],new]):
                sql=_sql(n); score=strength+1.4*(current==c.lower())-.12*i-.1*j+.2*sum(w in q for w in re.findall('[a-z]+',c.lower()) if len(w)>2)
                out[sql]=max(out.get(sql,-100),score)
    return sorted(out,key=lambda sql:-out[sql])
