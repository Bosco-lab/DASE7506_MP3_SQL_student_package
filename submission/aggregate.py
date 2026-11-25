"""Use explicit question references to distinguish measures, counts, and grouping entities."""
import re
from sqlglot import exp
from extra_repairs import _column, _parse, _sql, _tables, _set


def terms(text):
    return {w.removesuffix('s') for w in re.findall(r'[a-z]+',text.lower()) if len(w)>2}


def repairs(tree, question, schema):
    group = tree.args.get('group')
    order = tree.args.get('order')
    sources = _tables(tree)
    if not group or tree.find(exp.Subquery) or not sources:
        return
    # Superlative over a named measure stored on each row: SUM(measure), not COUNT(rows).
    if order and tree.args.get('limit') and len(sources)==1:
        target = re.search(r'\b(?:most|fewest|least|greatest|largest)\s+(\w+)',question,re.I)
        sort_expr = order.expressions[0].this
        if target and isinstance(sort_expr,exp.Count):
            table=sources[0];meta=schema.get(table.name.lower(),{}); noun=terms(target.group(1))
            for col in meta.get('columns',{}).values():
                if terms(col)==noun and noun and not noun.intersection({'name','id','type','number','kind','common','frequent'}):
                    seq=order.copy();seq.expressions[0].set('this',exp.Sum(this=_column(col,table.alias_or_name)))
                    yield 'measure_sum',7.,_set(tree,order=seq)
    # A request for the name and frequency of an entity gives a projected GROUP BY key.
    if len(tree.expressions)==1 and len(group.expressions)==1 and tree.args.get('having'):
        projected=tree.expressions[0].unalias();key=group.expressions[0]
        if isinstance(projected,exp.Column) and isinstance(key,exp.Column) and 'name' in terms(projected.name) and re.search(r'\bnames?\b',question,re.I) and 'id' in key.name.lower():
            if not (projected.table and key.table and projected.table.lower()!=key.table.lower()):
                for count in tree.args['having'].find_all(exp.Count):
                    if isinstance(count.this,exp.Distinct) and not re.search(r'\bdistinct|\bdifferent|\bunique',question,re.I):
                        out=_set(tree,group=exp.Group(expressions=[projected.copy()]))
                        next(out.args['having'].find_all(exp.Count)).set('this',exp.Star())
                        yield 'entity_count',5.,out
    # Count rows of the child relation when a single dimension table cannot supply the requested frequency.
    if len(sources)==1 and tree.find(exp.Count) and (order or tree.args.get('having') or re.search(r'\bnumber of\b',question,re.I)):
        parent=sources[0];candidates=[];q=terms(question)
        for name,meta in schema.items():
            if name==parent.name.lower(): continue
            for fk in meta['foreign_keys']:
                if fk['table'].lower()!=parent.name.lower():continue
                child_words=terms(name)-terms(parent.name)
                evidence=child_words&q
                if not evidence:continue
                child=exp.to_table(meta['name'],quoted=True)
                on=exp.EQ(this=_column(fk['from'],child.alias_or_name),expression=_column(fk['to'],parent.alias_or_name))
                augmented=_set(tree,joins=[exp.Join(this=child,on=on)])
                candidates.append((len(evidence),augmented))
        if len(candidates)==1:
            yield 'count_related',6.,candidates[0][1]
    # If asked how many times an event occurred, the multiplicity is a count rather than a SUM.
    if order and tree.args.get('limit') and re.search(r'\b(number of times|most often|most frequently)\b',question,re.I):
        for agg in order.find_all(exp.Sum):
            new=order.copy();next(new.find_all(exp.Sum)).replace(exp.Count(this=exp.Star()))
            yield 'frequency_count',8.,_set(tree,order=new)
            break


def propose(obs,base,info):
    checked=[e for e in obs['history'] if e['tool']=='check']; passing=[e for e in checked if e['observation']['matches_example']]; out={}
    for i,sql in enumerate(dict.fromkeys([base,*[e['sql'] for e in (passing or checked)]])):
        tree=_parse(sql)
        if not isinstance(tree,exp.Select):continue
        for kind,strength,new in repairs(tree,obs['task']['question'],info['tables']):
            text=_sql(new);value=strength-.2*i
            if text not in out or value>out[text][0]:out[text]=(value,kind,text)
    return list(out.values())
