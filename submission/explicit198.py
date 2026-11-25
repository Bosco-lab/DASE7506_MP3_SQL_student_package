"""Turn explicit language and public result headers into a high-confidence check."""
import re
from sqlglot import exp
from extra_repairs import _column, _headers, _parse, _set, _sql, _tables
from ground198 import simple_listing, aligned_union, words
from scope_repair import edits as filter_edits
from parent_agent import sql_key


def alphabetical(tree, q, info):
    order=tree.args.get('order')
    if 'alphabet' not in q or not order or len(order.expressions)!=1: return
    col=order.expressions[0].this
    if not isinstance(col,exp.Column) or 'id' not in col.name.lower():return
    raw={t['name'].lower():t for t in info['raw']}
    tables=_tables(tree)
    for owner in tables:
        if col.table and col.table.lower()!=owner.alias_or_name.lower():continue
        for fk in raw.get(owner.name.lower(),{}).get('foreign_keys',[]):
            if fk['from'].lower()!=col.name.lower():continue
            target=raw.get(fk['table'].lower());
            if not target:continue
            descriptions=[c['name'] for c in target['columns'] if any(k in c['type'].lower() for k in ('char','text','clob'))]
            if len(descriptions)!=1:continue
            ref=next((t for t in tables if t.name.lower()==target['name'].lower()),None)
            updated=tree.copy()
            if ref is None:
                ref=exp.to_table(target['name'],quoted=True)
                updated=updated.join(ref.copy(),on=exp.EQ(this=col.copy(),expression=_column(fk['to'],ref.alias_or_name)))
            updated.args['order'].expressions[0].set('this',_column(descriptions[0],ref.alias_or_name))
            yield 'text_order',updated


def count_header(tree,obs,info):
    headers=obs['example']['columns'];q=obs['task']['question'].lower()
    if len(headers)!=1 or headers[0].lower()!='count(*)' or tree.find(exp.Subquery) or len(tree.expressions)!=1:return
    agg=tree.expressions[0].unalias()
    if not isinstance(agg,exp.Count):return
    if tree.args.get('group') or re.search(r"\b(distinct|different|unique|no|not|without|never)\b|n't\b",q):return
    if not isinstance(agg.this,exp.Star):
        fixed=tree.copy();c=fixed.expressions[0].find(exp.Count);c.set('this',exp.Star());fixed.set('distinct',None)
        joins=fixed.args.get('joins') or []
        if any(j.side=='LEFT' for j in joins) and not re.search(r'\b(all|no|without|not|including|never)\b',q):
            for j in joins:j.set('side',None)
        yield 'row_count_header',fixed
    # Counting an entity type in a reference table: a missing join cannot create the requested count.
    if tree.args.get('where') or tree.args.get('joins'):return
    match=re.search(r'\bhow many (.+?) (?:do|are|exist|have)\b',q)
    if not match:return
    target=words(match[1]);existing=words(_tables(tree)[0].name) if _tables(tree) else set()
    if target&existing:return
    wanted=target-{'type','kind','different','distinct'}
    if not wanted:return
    choices=[t['name'] for t in info['raw'] if wanted<=words(t['name']) and ('ref' in words(t['name']) or 'type' in words(t['name']))]
    if len(choices)==1:yield 'count_entity',exp.select(exp.Count(this=exp.Star())).from_(exp.to_table(choices[0],quoted=True))


def explicit_options(obs,info):
    q=obs['task']['question'].lower();checks=[e for e in obs['history'] if e['tool']=='check']; hints=next((e['observation'] for e in obs['history'] if e['tool']=='repair_hint'),[])
    checked={sql_key(e['sql']):e['observation']['matches_example'] for e in checks};out={}
    passing=[e['sql'] for e in checks if e['observation']['matches_example']]
    for seed in dict.fromkeys([*passing,*hints,obs['issue_sql']]):
        t=_parse(seed)
        if not isinstance(t,exp.Select):continue
        for k,node in [*alphabetical(t,q,info),*simple_listing(t,obs,info),*aligned_union(t,obs,info),*filter_edits(t,q),*count_header(t,obs,info)]:
            s=_sql(node);key=sql_key(s)
            if key!=sql_key(seed) and checked.get(key) is not False:out.setdefault(key,(k,s))
    return list(out.values())
