"""One contrastive EXECUTE on the episode's public database, at local06 submit time."""
from dataclasses import dataclass
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError


def quote(s):
    return '"' + s.replace('"', '""') + '"'


def render(e):
    return e.sql(dialect='sqlite')


def parse(sql):
    try:
        return sqlglot.parse_one(sql, read='sqlite')
    except (ParseError, TokenError, RecursionError):
        return None


def flat(select):
    return isinstance(select, exp.Select) and not select.find(exp.Subquery, exp.CTE, exp.SetOperation)


def scope(select):
    return {t.alias_or_name.lower(): t.name.lower() for t in select.find_all(exp.Table)}


def field(col, select, schema):
    if not isinstance(col, exp.Column):
        return None
    refs = scope(select)
    matches = [t for alias, t in refs.items() if (not col.table or col.table.lower() == alias)
               and col.name.lower() in schema.get(t, {})]
    if len(matches) == 1:
        return matches[0], col.name.lower()


def tokens(text):
    return set(re.findall('[a-z]+', re.sub('([a-z])([A-Z])', r'\1 \2', text).lower()))


def conjunction(where):
    if isinstance(where, exp.And):
        yield from conjunction(where.this)
        yield from conjunction(where.expression)
    else:
        yield where


def equalities(select, schema):
    """Simple conjunctive filters. Reject partial or ambiguous parses."""
    if not select.args.get('where'):
        return {}
    result = {}
    for term in conjunction(select.args['where'].this):
        if not isinstance(term, exp.EQ):
            return {}
        a, b = term.this, term.expression
        if isinstance(a, exp.Lower) and isinstance(b, exp.Lower):
            a, b = a.this, b.this
        key = field(a, select, schema)
        if not key or not isinstance(b, exp.Literal) or not b.is_string or key in result:
            return {}
        result[key] = b.name
    return result


@dataclass(frozen=True)
class Plan:
    kind: str
    query: str
    rival: str
    refinement: str = ''

    def supports(self, row):
        if not row or any(type(x) is not int or x < 0 for x in row):
            return False
        if self.kind == 'filter':
            # Rival's named domain exists, but the question's literal is absent.
            # Incumbent matches a different native domain and loses that support downstream.
            return len(row) == 4 and row[0] >= row[1] > 0 and row[2] > 0 and row[3] == 0
        if self.kind == 'intersection':
            return len(row) == (4 if self.refinement else 2) and max(row[:2]) > 0 and min(row[:2]) == 0
        if self.kind == 'anti_fk':
            # Both anti-joins are vacuous on an empty child but the parent is populated.
            return len(row) == 2 and row[0] > 0 and row[1] == 0
        if self.kind == 'fanout':
            # Multiple qualifying-side rows per parent in the join without its filter.
            return len(row) == 3 and row[0] > row[1] > 0 and row[1] == row[2]
        return False


def filter_plan(base, alt, rival, q, schema):
    if not flat(base) or not flat(alt) or set(scope(base).values()) != set(scope(alt).values()):
        return
    if [field(e.unalias(), base, schema) for e in base.expressions] != [field(e.unalias(), alt, schema) for e in alt.expressions]:
        return
    # Require the same join structure and projection; ignore identifier case/quoting.
    def skeleton(tree):
        tree = tree.copy(); tree.set('where', None)
        for i in tree.find_all(exp.Identifier):
            i.set('quoted', False); i.set('this', i.name.lower())
        return render(tree)
    if skeleton(base) != skeleton(alt):
        return
    left, right = equalities(base, schema), equalities(alt, schema)
    only_l, only_r = set(left.items())-set(right.items()), set(right.items())-set(left.items())
    if len(only_l) != 1 or len(only_r) != 1:
        return
    ((lt, lc), lv), = only_l
    ((rt, rc), rv), = only_r
    # A uniquely question-grounded alternative, independent of the public answer cells.
    compact = re.sub('[^a-z0-9]', '', q)
    named = lambda c: len(c.replace('_', '')) >= 3 and c.replace('_', '') in compact
    if not named(rc) or rv.casefold() not in q or (named(lc) and lv.casefold() in q) or (lt,lc)==(rt,rc):
        return
    def marginal(t, c, v):
        ident = quote(c); tab = quote(t); literal = render(exp.Literal.string(v))
        return [f'(SELECT COUNT({ident}) FROM {tab})',
                f'(SELECT COUNT(*) FROM {tab} WHERE LOWER({ident}) = LOWER({literal}))']
    return Plan('filter', 'SELECT ' + ', '.join(marginal(lt,lc,lv)+marginal(rt,rc,rv)), rival)


def intersection_plan(base, alt, rival, q, schema):
    if not flat(base) or not isinstance(alt, exp.Intersect) or 'both' not in tokens(q):
        return
    sides = (alt.this, alt.expression)
    if not all(flat(s) and s.args.get('where') and len(s.expressions)==1 for s in sides):
        return
    projections = [field(s.expressions[0].unalias(), s, schema) for s in sides]
    if projections[0] is None or projections[0] != projections[1] or len(base.expressions)!=1:
        return
    if field(base.expressions[0].unalias(), base, schema) != projections[0]:
        return
    ranges = []; counts = []; clean = []
    for side in sides:
        pred = side.args['where'].this
        if not isinstance(pred, (exp.GT, exp.LT)) or not isinstance(pred.expression, exp.Literal):
            return
        fc = field(pred.this, side, schema); n = pred.expression.name
        if not fc or not re.fullmatch(r'\d+', n) or not re.search(r'(?<!\d)'+n+r'(?!\d)', q):
            return
        words = r'older than|above|greater than|over|more than' if isinstance(pred,exp.GT) else r'younger than|below|less than|under|lower than'
        if not re.search(r'\b(?:'+words+r')\s+'+n+r'\b', q):
            return
        ranges.append((fc,type(pred)))
        tab, col = fc; op = '>' if isinstance(pred, exp.GT) else '<'
        clause = f'FROM {quote(tab)} WHERE {quote(col)} {op} {render(pred.expression)}'
        counts.append(f'(SELECT COUNT(*) {clause})')
        if tab == projections[0][0]:
            clean.append(f'SELECT {quote(projections[0][1])} {clause}')
    if ranges[0][0]!=ranges[1][0] or ranges[0][1] == ranges[1][1]:
        return
    # If the rival carries an unasked-for join into BOTH branches, also check a
    # source-only intersection. Eligibility requires every referenced field locally.
    refine = ''
    extra = set(scope(sides[0]).values())-{projections[0][0]}
    if (len(clean)==2 and extra and not alt.args.get('order') and all(flat(s) and not any(s.args.get(k) for k in ('group','having','limit','order')) for s in sides)
            and set(scope(sides[0]).values()) == set(scope(sides[1]).values())
            and not any(tokens(t) & tokens(q) for t in extra)):
        refine = ' INTERSECT '.join(clean)
        for side in sides:
            joined = side.copy()
            joined.set('distinct', None)
            joined.set('expressions', [exp.Count(this=exp.Star())])
            counts.append(f'({render(joined)})')
    return Plan('intersection', 'SELECT '+', '.join(counts), rival, refine)


def anti_plan(base, alt, rival, q, schema, fks):
    if not isinstance(base,exp.Select) or not flat(alt) or not re.search(r"\b(no|never|without|not|don't|doesn't)\b",q):
        return
    where = base.args.get('where'); term = where.this if where else None
    if not isinstance(term, exp.Not) or not isinstance(term.this, exp.In):
        return
    isin=term.this; query=isin.args.get('query'); child=query.this if isinstance(query,exp.Subquery) else None
    if not flat(child) or len(child.expressions)!=1 or len(scope(child))!=1:
        return
    if not base.args.get('from') or not isinstance(base.args['from'].this, exp.Table) or base.args.get('joins'):
        return
    parent=base.args['from'].this.name.lower(); wrong=field(child.expressions[0],child,schema)
    if not wrong or not isinstance(isin.this,exp.Column): return
    key=(parent,isin.this.name.lower()); joins=alt.args.get('joins') or []
    if len(joins)!=1 or joins[0].side!='LEFT' or not isinstance(joins[0].args.get('on'),exp.EQ): return
    on=joins[0].args['on']; pair=[field(e,alt,schema) for e in (on.this,on.expression)]
    null=alt.args.get('where'); null=null.this if null else None
    if not isinstance(null,exp.Is) or not isinstance(null.expression,exp.Null): return
    ref=field(null.this,alt,schema)
    if (not ref or ref[0]!=wrong[0] or ref==wrong or (ref,key) not in fks or (wrong,key) in fks
            or key not in pair or ref not in pair or joins[0].this.name.lower()!=ref[0]): return
    if len(base.expressions)!=1 or len(alt.expressions)!=1 or field(alt.expressions[0],alt,schema)!=key or not isinstance(base.expressions[0],exp.Column) or base.expressions[0].name.lower()!=key[1]: return
    return Plan('anti_fk', f'SELECT (SELECT COUNT({quote(key[1])}) FROM {quote(parent)}), (SELECT COUNT(*) FROM {quote(ref[0])})', rival)


def fanout_plan(base, alt, rival, q, schema, fks):
    if not flat(base) or not flat(alt) or base.args.get('distinct') or not alt.args.get('distinct'):
        return
    bare=alt.copy(); bare.set('distinct',None)
    if bare!=base or len(base.expressions)!=1 or not base.args.get('where') or not re.search(r'\b(who|which|that)\b.*\b(have|has|with)\b',q): return
    if any(base.args.get(k) for k in ('group','limit','having','order')): return
    joins=base.args.get('joins') or []
    if len(joins)!=1 or joins[0].side or not isinstance(joins[0].args.get('on'),exp.EQ): return
    on=joins[0].args['on']; pair=[field(c,base,schema) for c in (on.this,on.expression)]
    proj=field(base.expressions[0],base,schema)
    if not proj or not all(pair): return
    for child,parent in (pair,pair[::-1]):
        if (child,parent) not in fks or parent[0]!=proj[0] or schema[parent[0]][parent[1]]['pk']!=1: continue
        if sum(bool(c['pk']) for c in schema[parent[0]].values())!=1: continue
        if not all((field(c,base,schema) or ('',))[0]==child[0] for c in base.args['where'].find_all(exp.Column)): continue
        parent_alias=next(k for k,v in scope(base).items() if v==parent[0]); key=exp.column(parent[1],parent_alias,quoted=True)
        probe=base.copy(); probe.set('where',None)
        probe.set('expressions',[exp.Count(this=exp.Star()),exp.Count(this=exp.Distinct(expressions=[key])),exp.Count(this=exp.Distinct(expressions=[base.expressions[0].copy()]))])
        return Plan('fanout',render(probe),rival)


def plan(base_sql, obs):
    if obs['remaining']<1: return
    history=obs['history']; passed=[e['sql'] for e in history if e['tool']=='check' and e['observation']['matches_example']]
    if base_sql not in passed or len(set(passed))<2: return
    raw=next((e['observation'] for e in history if e['tool']=='schema'),None)
    if not raw: return
    schema={t['name'].lower():{c['name'].lower():c for c in t['columns']} for t in raw}
    fks={((t['name'].lower(),k['from'].lower()),(k['table'].lower(),k['to'].lower())) for t in raw for k in t['foreign_keys'] if k.get('to')}
    q=obs['task']['question'].casefold(); base=parse(base_sql); candidates=[]
    for rival in dict.fromkeys(passed):
        if rival==base_sql: continue
        alt=parse(rival)
        for fn, args in ((anti_plan,(fks,)),(intersection_plan,()),(filter_plan,()),(fanout_plan,(fks,))):
            if fn is filter_plan and obs['example']['row_count']!=0: continue
            p=fn(base,alt,rival,q,schema,*args)
            if p: candidates.append(p)
    if candidates and all(p.kind==candidates[0].kind and p.query==candidates[0].query for p in candidates):
        p=candidates[0]
        if len(p.query.encode())<=16000 and not any(e['tool']=='execute' and e['sql']==p.query for e in history): return p


class Witness:
    def __init__(self, base, obs):
        self.base=base; self.plan=plan(base,obs); self.phase='start'

    def act(self,obs):
        p=self.plan
        if not p: return {'tool':'submit','sql':self.base}
        if self.phase=='start':
            self.phase='evidence'; return {'tool':'execute','sql':p.query}
        last=obs['history'][-1]; result=last['observation']
        if self.phase=='check':
            ok=(last['tool']=='check' and last['sql']==p.refinement and result['matches_example'])
            return {'tool':'submit','sql':p.refinement if ok else p.rival}
        if (last['tool']!='execute' or last['sql']!=p.query or not result['ok'] or result['truncated']
                or result['row_count']!=1 or not p.supports(result['rows'][0])):
            return {'tool':'submit','sql':self.base}
        if p.refinement and obs['remaining'] and result['rows'][0][2:] == [0, 0]:
            prev=next((e for e in obs['history'] if e['tool']=='check' and e['sql']==p.refinement),None)
            if prev is None:
                self.phase='check'; return {'tool':'check','sql':p.refinement}
            if prev['observation']['matches_example']: return {'tool':'submit','sql':p.refinement}
        return {'tool':'submit','sql':p.rival}
