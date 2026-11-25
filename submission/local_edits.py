"""Repair a scoped SQL defect, leaving the remainder of the query intact."""
import re
from sqlglot import exp
from extra_repairs import _parse, _sql, _tables, _column, _headers, _set
from collections import deque


def shortest(info,start,end):
    pending=deque([(start,[])])
    while pending:
        table,path=pending.popleft()
        if table==end:return path
        if len(path)>=3:continue
        for e in info['fk_graph'].get(table,[]):
            dest=e['to_table'].lower()
            if dest!=start and all(edge['to_table'].lower()!=dest for edge in path):pending.append((dest,path+[e]))
    return []


def grounded_join(tree,info):
    tables=_tables(tree);by_alias={t.alias_or_name.lower():t for t in tables}
    joins=tree.args.get('joins') or []
    for i,j in enumerate(joins):
        on=j.args.get('on');target=j.this
        if j.side or not isinstance(target,exp.Table) or not isinstance(on,exp.EQ):continue
        left,right=on.this,on.expression
        if not isinstance(left,exp.Column) or not isinstance(right,exp.Column):continue
        if left.table.lower()==target.alias_or_name.lower():left,right=right,left
        parent=by_alias.get(left.table.lower());
        if parent is None or right.table.lower()!=target.alias_or_name.lower():continue
        edges=info['fk_graph'].get(parent.name.lower(),[])
        if any(e['to_table'].lower()==target.name.lower() and e['from_col'].lower()==left.name.lower() and e['to_col'].lower()==right.name.lower() for e in edges):continue
        path=shortest(info,parent.name.lower(),target.name.lower())
        if not path or any(e['to_table'].lower() in {t.name.lower() for t in tables} for e in path[:-1]):continue
        # Equal names can be undeclared relationships. Require stronger evidence to replace them.
        if left.name.lower()==right.name.lower() and len(path)>1:continue
        replacement=[];src=parent
        for pos,e in enumerate(path):
            dest=target if pos==len(path)-1 else exp.to_table(e['to_table'],quoted=True)
            test=exp.EQ(this=_column(e['from_col'],src.alias_or_name),expression=_column(e['to_col'],dest.alias_or_name));replacement.append(exp.Join(this=dest.copy(),on=test));src=dest
        yield 'fk_path',8.,_set(tree,joins=[*joins[:i],*replacement,*joins[i+1:]])


def revisions(tree,obs,info):
    question=obs['task']['question'];q=question.casefold();schema=info['tables'];tables=_tables(tree)
    if not tables:return
    if not tree.find(exp.Subquery):yield from grounded_join(tree,info)
    # A full name bound to a surname field needs two predicates on the same entity.
    for eq in tree.find_all(exp.EQ):
        col,lit=eq.this,eq.expression
        if not isinstance(col,exp.Column) or not isinstance(lit,exp.Literal) or not lit.is_string:continue
        parts=lit.name.split(); short=re.sub('[^a-z]','',col.name.lower()); select=eq.find_ancestor(exp.Select)
        if len(parts)!=2 or lit.name.casefold() not in q or not select or short not in {'lname','lastname','surname','fname','firstname','givenname'}:continue
        for t in _tables(select):
            if col.table and col.table.casefold()!=t.alias_or_name.casefold():continue
            names={re.sub('[^a-z]','',c):v for c,v in schema.get(t.name.lower(),{}).get('columns',{}).items()}
            fn=next((names[k] for k in ('firstname','fname','givenname') if k in names),None);ln=next((names[k] for k in ('lastname','lname','surname') if k in names),None)
            if not(fn and ln):continue
            new=tree.copy(); match=next(e for e in new.find_all(exp.EQ) if _sql(e)==_sql(eq));match.replace(exp.and_(exp.EQ(this=_column(fn,t.alias_or_name),expression=exp.Literal.string(parts[0])),exp.EQ(this=_column(ln,t.alias_or_name),expression=exp.Literal.string(parts[1]))));yield 'name',10.,new
    group=tree.args.get('group');order=tree.args.get('order');having=tree.args.get('having')
    if group:
        for c in tree.find_all(exp.Count):
            if c.find_ancestor(exp.Select) is not tree or not c.find_ancestor(exp.Order,exp.Having) or re.search(r'\bdistinct|\bdifferent|\bunique',q):continue
            if not isinstance(c.this,exp.Distinct):continue
            new=tree.copy();m=next(e for e in new.find_all(exp.Count) if _sql(e)==_sql(c));m.set('this',exp.Star());yield 'count_rows',4.4,new
        if order and tree.args.get('limit') and len(group.expressions)==1:
            g=group.expressions[0]
            if isinstance(g,exp.Column) and tree.find(exp.Count) and re.search(r'\b(most|least) (?:common|frequent|popular)\b',q):
                for t in tables:
                    meta=schema.get(t.name.lower(),{});pks=meta.get('pks',[])
                    if g.table and g.table.lower()!=t.alias_or_name.lower():continue
                    if g.name.lower() not in meta.get('columns',{}) or len(pks)!=1 or pks[0].lower()==g.name.lower():continue
                    yield 'group_key',4.5,_set(tree,group=exp.Group(expressions=[_column(pks[0],t.alias_or_name)]))
    # Use public output expressions, including repeated columns and count forms, after a structural fix.
    if order and not group and tree.args.get('limit') and re.search(r'\bfirst\b',q):
        where=tree.args.get('where')
        if where and isinstance(where.this,exp.EQ) and where.this.find(exp.Subquery) and where.this.find(exp.Max,exp.Min) and isinstance(where.this.this,exp.Column):
            order_by=[exp.Ordered(this=where.this.this.copy(),desc=bool(where.this.find(exp.Max))),*[v.copy() for v in order.expressions]]
            yield 'ordered_extreme',4.5,_set(tree,where=None,order=exp.Order(expressions=order_by))


def proposals(obs,base,info):
    checks=[e for e in obs['history'] if e['tool']=='check'];passed={e['sql'] for e in checks if e['observation']['matches_example']};others=[e['sql'] for e in checks if e['sql'] in passed] if passed else [e['sql'] for e in reversed(checks)];out={}
    for i,sql in enumerate(dict.fromkeys([base,*others])):
        tree=_parse(sql)
        if not isinstance(tree,exp.Select):continue
        for kind,priority,new in revisions(tree,obs,info):
            text=_sql(new);score=priority-.18*i
            if text not in out or out[text][0]<score:out[text]=(score,kind,text)
    return sorted(out.values(),key=lambda x:-x[0])
