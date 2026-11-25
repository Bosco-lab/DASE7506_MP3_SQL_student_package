"""Schema-bound repair hypotheses. Every proposed rewrite must pass a public check."""
import re
from sqlglot import exp
from extra_repairs import _parse, _tables, _headers, _set, _sql


def risk(obs, passing, selected):
    if obs['example']['row_count'] > 1:
        return False
    tree = _parse(selected)
    return tree is not None and (bool(tree.find(exp.Subquery)) or
                    (obs['example']['row_count']==0 and bool(tree.find(exp.Join))))


def pass_quality(obs, candidate):
    if candidate['source']=='semantics':
        return 2.4
    if candidate['source']!='llm_lora':
        return 0.
    cols = [s.casefold() for s in candidate['result']['columns']]
    exact = cols == [s.casefold() for s in obs['example']['columns']]
    return 1.15 if exact and obs['example']['row_count']<=1 else .4


def semantic_proposals(obs, info, candidates):
    question=obs['task']['question'].casefold()
    schema=info['tables']; expected=obs['example']['columns']
    passed=any(c['result']['matches_example'] for c in candidates)
    for c in candidates:
        tree=_parse(c['sql'])
        if not isinstance(tree,exp.Select):
            continue
        tables=_tables(tree)
        # A top-k request constrains LIMIT. A count threshold also equal to k
        # needs independent support in the question.
        having=tree.args.get('having'); limit=tree.args.get('limit')
        if having and limit and isinstance(having.this,exp.GTE) and isinstance(having.this.this,exp.Count):
            k=limit.expression.name
            if having.this.expression.name==k and re.search(r'\btop\s+'+re.escape(k)+r'\b',question):
                yield _sql(_set(tree,having=None))
        if c['source'] not in ('hint','issue'):
            continue
        # Exact cardinality in a grouped count. Keep explicit lower/upper bounds intact.
        if c['result']['matches_example'] and not re.search(r'\bat least|\bor more|\bmore than|\bno fewer|\bat most',question):
            corrected=tree.copy(); changed=False
            numbers={'1':'one','2':'two','3':'three','4':'four','5':'five'}
            for op in list(corrected.find_all(exp.GTE)):
                n=op.expression.name
                if isinstance(op.this,exp.Count) and op.find_ancestor(exp.Having) and n in numbers and re.search(r'\b(?:'+n+'|'+numbers[n]+r')\s+\w+',question):
                    op.replace(exp.EQ(this=op.this.copy(),expression=op.expression.copy()));changed=True
            if changed: yield _sql(corrected)
        # Merge compatible repairs: bind the public projection on an inner FK
        # join when a left join count was too large on the public example.
        joins=tree.args.get('joins') or []
        if joins and tree.args.get('group') and any(j.side=='LEFT' for j in joins) and 'count(*)' in [s.lower() for s in expected] and not c['result']['matches_example']:
            projection=_headers(expected,tables,schema,tree.expressions)
            if projection:
                inner=[j.copy() for j in joins]
                for j in inner: j.set('side',None)
                yield _sql(_set(tree,joins=inner,expressions=projection[0],distinct=None))
        if not tables or passed:
            continue
        projection=_headers(expected,tables,schema,tree.expressions)
        if not projection:
            continue
        if len(tables)==1 and tree.find(exp.Avg) and tree.args.get('having'):
            names=schema.get(tables[0].name.lower(),{}).get('columns',{})
            words=set(re.findall(r'\w+',question))
            for col in tree.args['having'].find_all(exp.Column):
                if col.name.lower() in names: continue
                parts=set(col.name.lower().split('_'))
                def fit(name):
                    components=set(name.split('_'))
                    return len(parts & components)+len(words & components)
                candidates_by_name=sorted(names,key=lambda n:-fit(n))
                if candidates_by_name and fit(candidates_by_name[0])>=3:
                    alt=candidates_by_name[0]
                    fixed=tree.transform(lambda e:exp.column(names[alt],quoted=True) if isinstance(e,exp.Column) and e.name.lower()==col.name.lower() else e)
                    yield _sql(_set(fixed,expressions=projection[0]))
        if len(tables)==1 and re.search(r'\b(above|below|higher|lower|older|younger|greater|less|more|larger|smaller)\b.{0,18}\baverage\b',question):
            measures=[x.this for a in candidates for node in [_parse(a['sql'])] if node for x in node.find_all(exp.Avg) if isinstance(x.this,exp.Column)]
            for col in measures:
                if col.name.lower() not in schema.get(tables[0].name.lower(),{}).get('columns',{}): continue
                comp=exp.LT if re.search(r'\b(below|lower|younger|less|smaller)\b',question) else exp.GT
                col=exp.column(col.name,quoted=True); avg=exp.select(exp.Avg(this=col.copy())).from_(tables[0].copy()).subquery()
                projected=projection[0]; groups=[x.copy() for x in projected if not x.find(exp.AggFunc)]
                result=exp.select(*projected).from_(tables[0].copy()).where(comp(this=col.copy(),expression=avg))
                if groups: result=result.group_by(*groups)
                yield _sql(result)
        # Logical negation across a relationship: subtract entities satisfying the positive predicate.
        where=tree.args.get('where')
        if where and re.search(r"\b(not|never|without|no)\b|n't\b",question) and not tree.args.get('group'):
            predicate=where.this
            if isinstance(predicate,exp.Not) and isinstance(predicate.this,exp.In) and tree.args.get('joins'):
                for t in tables:
                    projected=_headers(expected,[t],schema,[])
                    if projected:
                        lhs=exp.select(*projected[0]).from_(t.copy());rhs=_set(tree,where=exp.Where(this=predicate.this.copy()),expressions=projection[0])
                        yield _sql(exp.Except(this=lhs,expression=rhs,distinct=True))
            if isinstance(predicate,(exp.EQ,exp.NEQ)) and (isinstance(predicate,exp.NEQ) or 'never' in question):
                positive=exp.EQ(this=predicate.this.copy(),expression=predicate.expression.copy())
                for t in tables:
                    lhs_proj=_headers(expected,[t],schema,[])
                    if not lhs_proj: continue
                    lhs=exp.select(*lhs_proj[0]).from_(t.copy()); rhs=_set(tree,where=exp.Where(this=positive),expressions=projection[0],order=None,limit=None)
                    yield _sql(exp.Except(this=lhs,expression=rhs,distinct=True))
                    if positive.expression.is_string:
                        rhs=rhs.copy(); pred=rhs.args['where'].this
                        pred.set('this',exp.Lower(this=pred.this.copy()));pred.set('expression',exp.Lower(this=pred.expression.copy()))
                        yield _sql(exp.Except(this=lhs.copy(),expression=rhs,distinct=True))
