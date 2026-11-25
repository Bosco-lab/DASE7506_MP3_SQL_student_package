"""Small relational edits whose justification is visible in the current episode."""
import re
from sqlglot import exp
from extra_repairs import _column, _headers, _parse, _set, _sql, _tables
from parent_agent import legacy, sql_key
from scope_repair import edits as filter_edits


def words(s):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return {w[:-3]+'y' if w.endswith('ies') else w.removesuffix('s') for w in re.findall('[a-z]+', s.lower())}


def projected_headers(tree, obs, info):
    headers = obs['example']['columns']
    sources = _tables(tree)
    desired = _headers(headers, sources, info['tables'], tree.expressions)
    if not desired:
        return
    head = desired[0]
    if len(head) != len(tree.expressions):
        return
    def label(e):
        return _sql(e.unalias()).lower().replace('"','').split(' as ')[0]
    for h,old,new in zip(headers, tree.expressions, head):
        if isinstance(new, exp.AggFunc) and label(old) != label(new):
            corrected = []
            for e,n in zip(tree.expressions, head):
                value = n.copy() if isinstance(n,exp.AggFunc) else e.unalias().copy()
                if e.alias:
                    value = exp.alias_(value,e.alias,quoted=True)
                corrected.append(value)
            yield 'aggregate_header', _set(tree,expressions=corrected)
            break


def simple_listing(tree, obs, info):
    question = obs['task']['question'].lower().rstrip(' ?.'); headers = obs['example']['columns']
    # All requested attributes and entity names account for the entire question.
    tokens = words(question)-set('list show what which are is the all of and give me please return find ids id name names details for each their a an'.split())
    if obs['ordered'] or not re.match(r'^(list|show|what|which|give|return|find)\b', question):
        return
    for tb in _tables(tree):
        if not words(tb.name) & words(question):
            continue
        head = _headers(headers,[tb],info['tables'],tree.expressions)
        if not head or any(e.find(exp.AggFunc) for e in head[0]):
            continue
        if tokens - words(tb.name) - set().union(*(words(h) for h in headers)):
            continue
        if not any(tree.args.get(x) for x in ('where','group','joins','limit','order')):
            continue
        yield 'list_entity', exp.select(*head[0]).from_(tb.copy())


def aligned_union(tree, obs, info):
    question = obs['task']['question'].casefold()
    if ' and ' not in question or len(obs['example']['columns'])!=1 or tree.args.get('where') or tree.args.get('group'):
        return
    choices = []
    for t in _tables(tree):
        if not words(t.name) <= words(question):
            continue
        projected = _headers(obs['example']['columns'], [t], info['tables'], [])
        if projected and isinstance(projected[0][0],exp.Column):
            choices.append(exp.select(*projected[0]).from_(t.copy()))
    if len(choices)==2:
        yield 'shared_attribute', exp.Union(this=choices[0],expression=choices[1],distinct=True)


def parent_scope(tree,obs,info):
    joins = tree.args.get('joins') or []
    tables = _tables(tree)
    if not joins or len(tables)!=len(joins)+1 or tree.find(exp.Subquery,exp.CTE):
        return
    def owner(c):
        matches = [t for t in tables if (not c.table or c.table.casefold()==t.alias_or_name.casefold()) and c.name.casefold() in info['tables'].get(t.name.casefold(),{}).get('columns',{})]
        return matches[0] if len(matches)==1 else None
    q = words(obs['task']['question']); headers=obs['example']['columns']
    # If every returned attribute and predicate belongs to one entity, a join may invent duplicates.
    fields = [c for k,v in tree.args.items() if k not in ('from','from_','joins') for e in (v if isinstance(v,list) else [v]) if isinstance(e,exp.Expression) for c in e.find_all(exp.Column)]
    owners = [owner(c) for c in fields]
    if not owners or any(t is None for t in owners) or len(set(owners))!=1:
        return
    target=owners[0]
    if not words(target.name) & q or any(words(t.name)&q for t in tables if t!=target):
        return
    if tree.find(exp.Count) and not all('count' in h.lower() for h in headers):
        return
    yield 'single_entity', _set(tree, **{'from':exp.From(this=target.copy()),'joins':[]})


def edits(tree,obs,info):
    yield from simple_listing(tree,obs,info)
    yield from aligned_union(tree,obs,info)
    yield from filter_edits(tree,obs['task']['question'])
    yield from projected_headers(tree,obs,info)
    yield from parent_scope(tree,obs,info)


class Stage:
    def __init__(self,base,obs):
        self.base=base
        from choose198 import aligned
        self.pool=list(aligned(obs,base))
        self.checked={sql_key(e['sql']):e['observation']['matches_example'] for e in obs['history'] if e['tool']=='check'}
        raw=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        tree=_parse(base)
        if raw and isinstance(tree,exp.Select):
            info=legacy.parse_schema(raw)
            seen={sql_key(base)}
            for kind,node in edits(tree,obs,info):
                sql=_sql(node);key=sql_key(sql)
                if key not in seen:
                    self.pool.append((kind,sql));seen.add(key)

    def act(self,obs):
        if obs['history'] and obs['history'][-1]['tool']=='check':
            e=obs['history'][-1];self.checked[sql_key(e['sql'])]=e['observation']['matches_example']
        for _,sql in self.pool:
            state=self.checked.get(sql_key(sql))
            if state is True:return {'tool':'submit','sql':sql}
            if state is None and obs['remaining']:return {'tool':'check','sql':sql}
        return {'tool':'submit','sql':self.base}
