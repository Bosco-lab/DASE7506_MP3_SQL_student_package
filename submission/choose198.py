"""Compare public-passing rivals only when they resolve a concrete semantic defect."""
import re
from sqlglot import exp
from extra_repairs import _parse, _tables, _sql, _set


def words(text):
    return {w.removesuffix('s') for w in re.findall('[a-z]+', text.lower())}


def normalize(text):
    return re.sub(r'\w+\.|[^\w*()]', '', text.lower().replace('"',''))


def aligned(obs,base_sql):
    q=obs['task']['question'].lower();headers=obs['example']['columns'];base=_parse(base_sql)
    if not isinstance(base,exp.Select):return
    names = [normalize(h) for h in headers]
    candidates=[e for e in obs['history'] if e['tool']=='check' and e['observation']['matches_example']]
    hints=next((e['observation'] for e in obs['history'] if e['tool']=='repair_hint'),[])
    for event in candidates:
        alt_sql=event['sql'];alt=_parse(alt_sql)
        if alt_sql==base_sql or not isinstance(alt,exp.Select):continue
        # The public aggregate expression distinguishes two competing projected attributes.
        if len(names)==1 and re.match(r'(count|sum|avg|min|max)\(',names[0]) and not names[0].endswith('(*)'):
            if len(alt.expressions)==1 and normalize(_sql(alt.expressions[0].unalias()))==names[0] and normalize(_sql(base.expressions[0].unalias()))!=names[0]:
                yield 'bound_aggregate',alt_sql
        if alt_sql not in hints:continue
        # An explicit entity called/named 'X' supplies the owner of the comparison, not only the value.
        named=re.search(r"\b(\w+) (?:called|named) [\"']([^\"']+)[\"']",q)
        if named and base.find(exp.Subquery) and not alt.find(exp.Subquery):
            sources={t.alias_or_name.lower():words(t.name) for t in _tables(alt)}
            where=alt.args.get('where')
            if where and isinstance(where.this,exp.EQ):
                col,lit=where.this.this,where.this.expression
                if isinstance(col,exp.Column) and isinstance(lit,exp.Literal) and lit.name.lower()==named[2] and 'name' in words(col.name) and words(named[1]) & sources.get(col.table.lower(),set()):
                    yield 'literal_owner',alt_sql
        if re.search(r'\b(topic|about|contains?|containing)\b',q) and not base.find(exp.Like) and alt.find(exp.Like):
            pred=alt.find(exp.Like)
            if isinstance(pred.this,exp.Column) and pred.this.name.lower() in [h.lower() for h in headers] and isinstance(pred.expression,exp.Literal) and pred.expression.name.strip('%').lower() in q:
                yield 'search_text',alt_sql
        # If asking how many types an entity has, a reference table counts exactly its declared categories.
        if 'how many' in q and 'type' in q and len(_tables(alt))==1 and len(_tables(base))==1 and not alt.args.get('where') and alt.find(exp.Count):
            names_alt=words(_tables(alt)[0].name);names_base=words(_tables(base)[0].name)
            if names_alt & {'type','ref'} and (names_alt & words(q))-{'type','ref'} and not names_base & words(q):
                yield 'type_table',alt_sql
        # Preserve the checked multiplicity of a measured attribute when the question requests named categories.
        if base.args.get('group') and alt.args.get('group') and not re.search(r'\bdistinct|\bunique|\bdifferent',q):
            if re.search(r'\bat least|\bmore than|\bfewer than',q) and base.args.get('having') and alt.args.get('having'):
                if base.args['having'].find(exp.Distinct) and alt.args['having'].find(exp.Star) and [e.name.lower() for e in alt.expressions]==[e.name.lower() for e in base.expressions]:
                    yield 'relation_frequency',alt_sql
