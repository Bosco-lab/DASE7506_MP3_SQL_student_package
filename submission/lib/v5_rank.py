"""PLAN2 public-only candidate features and raw-logit inference (Python + SQLGlot)."""
from functools import lru_cache
import math
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


_INTENTS = ('distinct|different|unique', 'how many|count|number of', 'average|avg|mean',
            'sum|total', 'max|maximum|highest|largest|most|oldest|newest|longest',
            'min|minimum|lowest|smallest|least|youngest|shortest',
            'order|sort|alphabetical|ascending|descending', 'never|not|except|without|no',
            'both', 'or', 'at least', 'at most', 'each|every|per')
_AST_NAMES = 'tables joins left_joins non_fk unused subqueries distinct group having order limit except intersect union not_in not_exists is_null like neq eq and or aggregates projections width_match length self_eq quoted'.split()
_CROSS_NAMES = 'table_overlap word_overlap static_header distinct_agrees agg_agrees order_agrees neg_agrees src_hint src_issue src_template src_simp src_llm'.split()
_DIFF_NAMES = 'same_sql projection_only added_like fewer_joins fewer_tables fewer_subqueries dropped_distinct fewer_conditions dropped_limit dropped_group dropped_except added_in shorter'.split()
_FAMILIES = ('seed proj qual simp name tmpl agg basic simp2 hint4 schema_fix align fk_join '
             'feedback_error feedback_projection feedback_semantic issue other').split()
_META_NAMES = ('rank_in_pool log_rank is_first_hint same_as_first_hint is_issue hint_votes base_issue base_first_hint base_alt_hint example_rows example_empty example_single unknown_fraction parse_valid fk_equations non_fk_equations'.split()
               + ['family_' + f for f in _FAMILIES])
_RED_NAMES = '''requested_distinct missing_distinct unrequested_distinct agg_present agg_missing unused_join suspicious_join flat_negation empty_lookup self_equality duplicate_binding ordered_present ordered_missing split_name full_name_literal swapped_name suspect_predicate duplicate_predicate any_exists exact_one copied_cell stopword_like join_without_on'''.split()
_EXEC_NAMES = 'exec_header_exact exec_header_overlap exec_type_match exec_row_count'.split()
FEATURE_NAMES = (['q_' + s.split('|')[0].replace(' ', '_') for s in _INTENTS] + ['ordered']
                 + _AST_NAMES + _CROSS_NAMES + _DIFF_NAMES + _META_NAMES
                 + ['red_' + s for s in _RED_NAMES] + _EXEC_NAMES)


def _norm(s):
    return ' '.join(s.lower().strip().rstrip(';').split())


@lru_cache(maxsize=256)
def _analysis(sql):
    """Bounded syntax-only cache. No task/schema/check state is stored."""
    try:
        tree = sqlglot.parse_one(sql, read='sqlite')
    except (SqlglotError, RecursionError):
        tree = None
    root = tree
    while isinstance(root, exp.SetOperation):
        root = root.left
    valid = isinstance(root, exp.Select)
    nodes = list(tree.walk()) if valid else []
    def count(cls):
        return sum(isinstance(n, cls) for n in nodes)
    def has(cls):
        return any(isinstance(n, cls) for n in nodes)
    where = root.args.get('where') if valid else None
    wnodes = [n for w in nodes if isinstance(w, exp.Where) for n in w.walk()]
    joins = [n for n in nodes if isinstance(n, exp.Join)]
    tbls = [n for n in nodes if isinstance(n, exp.Table)]
    texts = {cls: has(cls) for cls in (exp.Group, exp.Having, exp.Limit, exp.Order, exp.Except,
                                        exp.Union, exp.Intersect, exp.Exists)}
    sets = {type(n.this) for n in nodes if isinstance(n, exp.Not)}
    form = _norm(tree.sql(dialect='sqlite')) if valid else _norm(sql)
    args = root.args if valid else {}
    return dict(valid=valid, root=root, nodes=nodes, tables=[t.name.lower() for t in tbls],
                tbls=tbls, joins=joins, left=sum(j.side.lower() == 'left' for j in joins),
                sub=max(count(exp.Select)-1, 0), distinct=bool(args.get('distinct')),
                texts=texts, nin=exp.In in sets, nex=exp.Exists in sets,
                null=any(isinstance(n, exp.Is) and isinstance(n.expression, exp.Null) for n in nodes),
                like=has(exp.Like), neq=has(exp.NEQ), eq=sum(isinstance(n, exp.EQ) for n in wnodes),
                ands=sum(isinstance(n, exp.And) for n in wnodes), ors=sum(isinstance(n, exp.Or) for n in wnodes),
                agg=count(exp.AggFunc), ins=count(exp.In), proj=root.expressions if valid else [],
                where=where, where_text=where.sql(dialect='sqlite').lower() if where else '', canon=form)


def _schema_stats(a, schema):
    tables = (schema or {}).get('tables', {})
    graph = (schema or {}).get('fk_graph', {})
    aliases = {t.alias_or_name.lower(): t.name.lower() for t in a['tbls']}
    derived = {n.alias_or_name.lower() for n in a['nodes'] if isinstance(n, (exp.CTE, exp.Subquery, exp.Alias))}
    unknown = sum(t not in tables and t not in derived for t in a['tables'])
    total = len(a['tables'])
    for c in (n for n in a['nodes'] if isinstance(n, exp.Column) and not n.is_star):
        qualifier, name = c.table.lower(), c.name.lower()
        refs = [aliases.get(qualifier, qualifier)] if qualifier else a['tables']
        total += 1
        if qualifier not in derived and name not in derived:
            unknown += not any(name in tables.get(t, {}).get('columns', {}) for t in refs)
    root = a['root'] if a['valid'] else None
    used = set()
    for arg in ('expressions', 'where', 'group', 'order', 'having'):
        part = root.args.get(arg) if root is not None else None
        for e in part if isinstance(part, list) else ([part] if part is not None else []):
            if e.is_star:
                used.update([e.table.lower()] if isinstance(e, exp.Column) else aliases)
            for c in e.find_all(exp.Column):
                if c.table:
                    used.add(c.table.lower())
                else:
                    used.update(k for k,t in aliases.items() if c.name.lower() in tables.get(t, {}).get('columns', {}))
    unused = sum(isinstance(j.this, exp.Table) and j.this.alias_or_name.lower() not in used
                 for j in (root.args.get('joins') or [])) if root is not None and tables else 0
    fk = non_fk = 0
    suspicious = self_eq = duplicate = missing_on = False
    seen = set()
    for j in a['joins']:
        on = j.args.get('on')
        if on is None:
            missing_on |= not bool(j.args.get('using'))
            continue
        for eq in on.find_all(exp.EQ):
            left, right = eq.this, eq.expression
            self_eq |= left == right
            if not isinstance(left, exp.Column) or not isinstance(right, exp.Column):
                continue
            lt, rt = (aliases.get(c.table.lower()) for c in (left, right))
            lc, rc = left.name.lower(), right.name.lower()
            if lt in tables and rt in tables and lt != rt:
                matched = any(e['from_col'].lower() == lc and e['to_table'].lower() == rt
                              and (e['to_col'] or '').lower() == rc for e in graph.get(lt, []))
                fk += matched
                non_fk += not matched
                suspicious |= not matched and lc != rc
            if isinstance(j.this, exp.Table):
                for target, other in ((left,right), (right,left)):
                    if target.table.lower() == j.this.alias_or_name.lower() and target.table.lower() != other.table.lower():
                        key = (j.this.name.lower(), other.table.lower(), other.name.lower())
                        duplicate |= key in seen
                        seen.add(key)
    return unknown/max(1,total) if tables else 0., unused, fk, non_fk, suspicious, self_eq, duplicate, missing_on


def _words(s):
    return set(re.findall('[a-z0-9]+', s.lower()))


def _overlap(a,b):
    a,b = _words(a), _words(b)
    return len(a & b)/max(1,len(a | b))


def _exec_features(example, check):
    if check is None:
        return [0.0] * len(_EXEC_NAMES)
    ec, gc = example['columns'], check.get('columns') or []
    exact = overlap = type_match = 0.
    er, gr = example.get('rows', []), check.get('rows') or []
    # Compare per-column type distributions, independent of row order and sample length.
    if len(ec) == len(gc) and ec:
        for j,(a,b) in enumerate(zip(ec,gc)):
            exact += a.casefold() == b.casefold()
            overlap += _overlap(a,b)
            if er and gr:
                from collections import Counter
                e = Counter(type(r[j]).__name__ for r in er)
                g = Counter(type(r[j]).__name__ for r in gr)
                type_match += sum(min(n/len(er), g[t]/len(gr)) for t,n in e.items())
        exact, overlap, type_match = (x/len(ec) for x in (exact, overlap, type_match))
    return [exact, overlap, type_match, check.get('row_count') or 0]


def features(obs, parsed_schema, hints, candidate, check=None):
    """One fixed-length vector. Priority omits check; choice passes its public result."""
    sql, issue = candidate['sql'], obs['issue_sql']
    a, b = _analysis(sql), _analysis(issue)
    q = (obs.get('task') or obs)['question']
    ex = obs['example']; cols = ex['columns']; rows = ex['row_count']
    intent = [bool(re.search(r'\b(?:' + pat + r')\b', q, re.I)) for pat in _INTENTS] + [bool(obs['ordered'])]
    unknown, unused, fk, nonfk, suspicious, selfeq, dup, no_on = _schema_stats(a, parsed_schema)
    t, bt = a['texts'], b['texts']
    exc = a['nin'] or a['nex'] or t[exp.Except] or a['null']
    width = len(a['proj']); width_ok = bool(cols) and width == len(cols)
    ast = [len(a['tbls']),len(a['joins']),a['left'],suspicious,unused,a['sub'],a['distinct'],t[exp.Group],
           t[exp.Having],t[exp.Order],t[exp.Limit],t[exp.Except],t[exp.Intersect],t[exp.Union],a['nin'],
           a['nex'],a['null'],a['like'],a['neq'],a['eq'],a['ands'],a['ors'],a['agg'],width,width_ok,
           min(len(sql)/300,3),selfeq,"'" in sql]
    src = candidate['source'].lower(); family = candidate.get('family', '').lower(); base = str(candidate.get('base', '')).lower()
    ql = q.lower(); qw = _words(q); tbls = a['tables']
    hdr = sum(max(_overlap(p.sql(),c), _overlap(p.alias_or_name,c)) for p,c in zip(a['proj'],cols))/width if width_ok else 0.
    cross = [sum(any(w in ql for w in n.split('_') if len(w)>=3) for n in tbls)/max(1,len(tbls)),
             len(qw & _words(sql))/max(1,len(qw)), hdr, intent[0] and a['distinct'],
             any(intent[1:4]) and a['agg']>0, intent[-1] and t[exp.Order],intent[7] and exc,
             'hint' in src,'issue' in src,'tmpl' in src or 'ast_from_hint' in src,'simp' in src and 'simp2' not in src,'llm' in src]
    same = a['canon'] == b['canon']
    diff = [same, not same and tbls==b['tables'] and len(a['joins'])==len(b['joins']) and a['eq']==b['eq'] and t[exp.Group]==bt[exp.Group] and t[exp.Order]==bt[exp.Order],
            a['like'] and not b['like'] and a['eq']<b['eq'], len(a['joins'])<len(b['joins']),len(tbls)<len(b['tables']),
            a['sub']<b['sub'],b['distinct'] and not a['distinct'],a['eq']+a['ands']<b['eq']+b['ands'],
            bt[exp.Limit] and not t[exp.Limit],bt[exp.Group] and not t[exp.Group],bt[exp.Except] and not t[exp.Except],
            a['ins']>b['ins'] or a['nin'] and not b['nin'], len(sql)<len(issue)]
    rank = candidate.get('rank_in_pool', 0)
    h0 = bool(hints) and a['canon']==_analysis(hints[0])['canon']
    votes = candidate.get('hint_votes')
    if votes is None:
        votes = sum(a['canon']==_analysis(h)['canon'] for h in hints)
    family_key = family if family in _FAMILIES else 'other'
    meta = [rank, math.log1p(rank), src in ('hint0','first_hint') or (src=='hint' and h0), h0, same, votes,
            'issue' in base, base=='hint0', 'hint' in base and base!='hint0',rows,rows==0,rows==1,unknown,a['valid'],fk,nonfk]
    meta.extend(family_key==k for k in _FAMILIES)
    w = a['where_text']; name = re.search(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', q)
    split = full = swapped = False
    if name:
        first,last = (x.lower() for x in name.groups())
        def name_eq(col,val):
            return bool(re.search(rf"\b{col}\s*=\s*'{val}'", w))
        split = name_eq('fname',first) and name_eq('lname',last)
        full = bool(re.search(rf"=\s*'{first}\s+{last}'", w))
        swapped = name_eq('fname',last) and name_eq('lname',first)
    eqs = [e.sql().lower() for e in a['where'].find_all(exp.EQ)] if a['where'] is not None else []
    cells = {c.strip().lower() for row in ex.get('rows',[]) for c in row if isinstance(c,str) and len(c.strip())>=3}
    literals = [n.this.strip().lower() for n in a['nodes'] if isinstance(n,exp.Literal) and n.is_string]
    agg_q = bool(re.search(r'\b(total|sum|average|avg|maximum|minimum|most|least|how many|number of|at least \d+|at most \d+)\b',ql))
    tables = (parsed_schema or {}).get('tables',{})
    lookup = (bool(tbls) and any('name' in c for c in tables.get(tbls[0],{}).get('columns',{}))
              and any(re.search('type|ref|category',n) for n in tbls[1:]) and bool(re.search(r'(typename|description|type_name|category_name)\s*=',w)))
    # Local sides of legacy peer-relative rules: the single-candidate API never reads
    # unobserved peer checks. P3 has no value or skeleton probe observations.
    red = [intent[0] and a['distinct'], intent[0] and not a['distinct'], not intent[0] and a['distinct'] and not t[exp.Group],
           agg_q and a['agg']>0,agg_q and not a['agg'],unused>0,suspicious,
           rows<=1 and intent[7] and bool(a['joins']) and not exc and bool(re.search('<>|!=|not like',w)),
           rows==0 and lookup,selfeq,intent[8] and dup,intent[-1] and t[exp.Order],intent[-1] and not t[exp.Order],
           split,full,swapped,'other_account_details' in w or 'or accounts.account_name = accounts.account_name' in w,
           len(eqs)>len(set(eqs)),bool(re.search(r'\b(after|before|greater than|less than)\s+any\b',ql)) and t[exp.Exists],
           bool(re.search(r'\b(two|2)\b.*\bor\s+(one|1)\b',ql)) and 'at least' not in ql and any(re.search(r'(?<![<>!])=\s*1\b',n.sql()) for n in a['nodes'] if isinstance(n,exp.Having)),
           any(s in cells and s not in ql+' '+issue.lower() for s in literals),
           any(isinstance(n,exp.Like) and isinstance(n.expression,exp.Literal) and re.search(r'\b(the|a|an)\b',str(n.expression.this),re.I) for n in a['nodes']),no_on]
    return list(map(float, intent + ast + cross + diff + meta + red + _exec_features(ex,check)))


def score(model, feats):
    """Raw pointwise logistic logit. Standardization is folded into weights/bias."""
    if len(feats) != len(model['weights']):
        raise ValueError('v5 feature/model dimension mismatch')
    return model['bias'] + sum(x*w for x,w in zip(feats, model['weights']))
