"""Terminal repair search using the remaining public checks and train-derived priors."""
from collections import Counter
import math
import re
from sqlglot import exp
from extra_repairs import _parse, _tables, _set, _sql, _headers


def words(text):
    return re.findall('[a-z]+|[0-9]+', text.lower())


def canon(text):
    return re.sub('[\\s`"\\[\\]]+', '', text.casefold())


def sqlnorm(sql):
    tree = _parse(sql)
    return _sql(tree) if tree is not None else sql.strip()


def distinct_features(question, tree, hints, issue):
    tokens = words(question)
    fs = {'bias', *('w='+w for w in tokens), *('pair='+a+'_'+b for a,b in zip(tokens,tokens[1:])), 'start='+('_'.join(tokens[:2]))}
    fs.add('join='+str(bool(tree.args.get('joins')))); fs.add('where='+str(bool(tree.args.get('where'))));fs.add('width='+str(min(len(tree.expressions),4)))
    for i,s in enumerate([issue,*hints[:3]]):
        tr = _parse(s)
        if isinstance(tr,exp.Select): fs.add(f'dist{i}='+str(bool(tr.args.get('distinct'))))
    for e in tree.expressions:
        if isinstance(e.unalias(),exp.Column):
            fs.update('proj='+w for w in words(e.unalias().name))
    return sorted(fs)


def distinct_logit(weights, fs):
    return sum(weights.get(f, 0.) for f in fs)


def edits(tree, obs, schema, hints, weights):
    question=obs['task']['question'].casefold(); expected=obs['example']['columns']; sources=_tables(tree)
    expressions=tree.expressions
    if not sources: return
    if not tree.args.get('group') and not tree.args.get('limit') and not any(e.find(exp.AggFunc) for e in expressions):
        z=distinct_logit(weights,distinct_features(question,tree,hints,obs['issue_sql']))
        if tree.args.get('distinct') and z<0:
            yield 'multiplicity', min(6.,-z), _set(tree,distinct=None)
        if not tree.args.get('distinct') and z>0:
            yield 'multiplicity', min(6.,z), _set(tree,distinct=exp.Distinct())
    # The public projection supplies both order and count-expression semantics.
    cols=lambda xs:[canon(_sql(e.unalias()).split('.')[-1]) for e in xs]
    for projection in _headers(expected,sources,schema,expressions)[:1]:
        diff=cols(projection)!=cols(expressions)
        if diff:
            p=cols(projection); o=cols(expressions)
            same_bag=Counter(p)==Counter(o)
            agg=any(e.find(exp.AggFunc) for e in projection+expressions)
            # Header binding must not silently turn a row projection into a global aggregate.
            if same_bag or (len(projection)==len(expressions) and (not agg or bool(tree.find(exp.AggFunc)))) or len(projection)!=len(expressions):
                yield 'projection', 9. if same_bag else 5.6, _set(tree,expressions=projection)
    where=tree.args.get('where'); group=tree.args.get('group'); order=tree.args.get('order')
    if isinstance(where,exp.Where) and isinstance(where.this,exp.And) and re.search(r'\bboth\b|\bas well as\b',question):
        a,b=where.this.this,where.this.expression
        if a.find(exp.Column) and b.find(exp.Column) and a.find(exp.Column).name.casefold()==b.find(exp.Column).name.casefold() and _sql(a)!=_sql(b):
            left=_set(tree,where=exp.Where(this=a));right=_set(tree,where=exp.Where(this=b))
            yield 'intersection',7.,exp.Intersect(this=left,expression=right,distinct=True)
    if re.search(r'\b(most|least) (?:common|popular|frequent)\b',question) and order and tree.args.get('limit') and len(expressions)==1 and not tree.find(exp.AggFunc):
        desc='least' not in question
        new=_set(tree,group=exp.Group(expressions=[expressions[0].unalias()]),order=exp.Order(expressions=[exp.Ordered(this=exp.Count(this=exp.Star()),desc=desc)]))
        yield 'frequency',7.,new
    # An empty anti-join should exclude related keys, not only keys of NULL-valued rows.
    for neg in tree.find_all(exp.Not):
        if isinstance(neg.this,exp.In) and re.search(r"\b(no|not|without|never)\b|n't",question):
            sub=neg.this.find(exp.Select)
            if sub and sub.args.get('where') and isinstance(sub.args['where'].this,exp.Is) and isinstance(sub.args['where'].this.expression,exp.Null) and 'null' not in question and 'missing' not in question:
                updated=tree.copy(); match=next((s for s in updated.find_all(exp.Select) if _sql(s)==_sql(sub)),None)
                if match is not None:
                    match.set('where',None);yield 'anti_scope',6.,updated
    # Restore case-insensitive text comparison only when the public value spelling is uncertain.
    if where:
        fixed=tree.copy(); changed=False
        for cmp in fixed.args['where'].find_all(exp.EQ,exp.NEQ):
            if isinstance(cmp.this,exp.Column) and isinstance(cmp.expression,exp.Literal) and cmp.expression.is_string and re.search('[A-Za-z]',cmp.expression.name):
                cmp.set('this',exp.Lower(this=cmp.this.copy()));cmp.set('expression',exp.Lower(this=cmp.expression.copy()));changed=True
        if changed:
            yield 'literal_case',2.1,fixed


def options(obs, base, schema, hints, weights):
    checks=[e for e in obs['history'] if e['tool']=='check'];by_sql={sqlnorm(e['sql']):e['observation'] for e in checks}
    example=obs['example']; base_result=by_sql.get(sqlnorm(base),{}); passing=base_result.get('matches_example',False)
    seeds=[base] if passing else [base,*hints,*reversed([e['sql'] for e in checks]),obs['issue_sql']]
    result={}
    for i,sql in enumerate(dict.fromkeys(seeds)):
        tree=_parse(sql)
        if not isinstance(tree,exp.Select): continue
        feedback=by_sql.get(sqlnorm(sql),{});headers=feedback.get('columns',[])
        header_match=bool(headers) and list(map(canon,headers))==list(map(canon,example['columns']))
        for kind,confidence,t in edits(tree,obs,schema,hints,weights):
            text=_sql(t); key=sqlnorm(text)
            if key==sqlnorm(base) or (key in by_sql and not by_sql[key]['matches_example']): continue
            if kind=='projection' and header_match:continue
            if passing and kind=='literal_case' and example['row_count']!=0: continue
            strength=confidence-.12*i
            if kind=='multiplicity' and feedback.get('ok') and not passing:
                wrong_sign=(feedback['row_count']<example['row_count'])==bool(t.args.get('distinct'))
                strength+= -5 if wrong_sign else 4
            if passing and kind=='multiplicity' and confidence<1.5:continue
            if key not in result or strength>result[key][0]:result[key]=(strength,kind,text)
    return sorted(result.values(),key=lambda item:-item[0])
