"""Correct a scalar extremum and its ordering using the observed query and output shape."""
import re
from sqlglot import exp
from witness import parse
from values_schema import words


def ranked_rows(obs):
    if not obs['ordered'] or obs['example']['row_count']!=1:
        return
    q=obs['task']['question'].lower()
    checks=[e for e in obs['history'] if e['tool']=='check']
    hints=next((e['observation'] for e in obs['history'] if e['tool']=='repair_hint'),[])
    for text in dict.fromkeys([*hints,*[e['sql'] for e in checks]]):
        tree=parse(text)
        if not isinstance(tree,exp.Select) or tree.args.get('group') or any(e.find(exp.AggFunc) for e in tree.expressions):continue
        wh=tree.args.get('where');cond=wh.this if wh else None
        if not isinstance(cond,exp.EQ) or not isinstance(cond.this,exp.Column) or not isinstance(cond.expression,exp.Subquery):continue
        sub=cond.expression.this
        if not isinstance(sub,exp.Select) or len(sub.expressions)!=1 or any(sub.args.get(k) for k in ('where','joins','group','having','limit')):continue
        agg=sub.expressions[0]; col=cond.this
        if not isinstance(agg,(exp.Min,exp.Max)) or not isinstance(agg.this,exp.Column) or agg.this.name.lower()!=col.name.lower():continue
        descending=isinstance(agg,exp.Max)
        if not set(words(col.name)) & set(words(q)) and not re.search(r'\boldest|youngest|cheapest|most expensive\b',q):continue
        updated=tree.copy();updated.set('where',None);updated.set('distinct',None);updated.set('order',None); key=col.sql(dialect='sqlite')
        tables=[tree.args['from'].this, *[j.this for j in tree.args.get('joins',[])]] if tree.args.get('from') else []
        if len(tables)>1 and all(isinstance(t,exp.Table) for t in tables):
            source=next((t for t in tables if t.alias_or_name.casefold()==col.table.casefold()),None)
            names={t.alias_or_name.casefold() for t in tables}
            if source and all(isinstance(e.unalias(),exp.Column) and e.unalias().table.casefold()==col.table.casefold() for e in tree.expressions):
                other_words=set().union(*(set(words(t.name))-set(words(source.name)) for t in tables if t!=source))
                if not other_words & set(words(q)):
                    updated.set('from',exp.From(this=source.copy())); updated.set('joins',[])
        # Re-check the natural direction before spending a call on its rival.
        yield 'single_extremum',updated.order_by(key+(' DESC' if descending else ' ASC')).limit(1).sql(dialect='sqlite')
        if not any(e['observation']['matches_example'] for e in checks):
            yield 'opposite_rank',updated.order_by(key+(' ASC' if descending else ' DESC')).limit(1).sql(dialect='sqlite')
