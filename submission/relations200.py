"""Public-observation relation rewrites for SQLite / sqlglot 27.29.

Call propose(obs, base_sql) on a pending submit, or on the final planned check
when the schema is visible and no check has matched. See RelationStage below.
There is no file access, model state, identifier allowlist or tool call here.
"""
from collections import defaultdict, deque
import re

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError


def parse(sql):
    if not sql or len(sql.encode()) > 16000:
        return None
    try:
        tree = parse_one(sql, read='sqlite')
        if isinstance(tree, exp.Select) and not tree.args.get('with') and not tree.find(exp.Window):
            return tree
    except (SqlglotError, RecursionError):
        pass


def sql(tree):
    return tree.sql(dialect='sqlite')


def words(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    return {re.sub(r'ies$', 'y', w) if w.endswith('ies') else w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w
            for w in re.findall('[a-z]+', text.lower())} - {'in', 'of', 'the', 'on', 'to', 'with', 'a', 'for'}


def scope(tree):
    root = tree.args.get('from')
    tables = [root.this] if root else []
    tables += [j.this for j in tree.args.get('joins', [])]
    if not all(isinstance(t, exp.Table) and isinstance(t.this, exp.Identifier) for t in tables):
        return {}
    return {t.alias_or_name.lower(): t.name.lower() for t in tables}


def column(alias, name):
    return exp.column(name, table=alias, quoted=True)


def table(name, alias=None):
    obj = exp.Table(this=exp.to_identifier(name, quoted=True))
    return obj.as_(alias, table=True) if alias else obj


class Graph:
    def __init__(self, schema):
        self.cols = {t['name'].lower(): {c['name'].lower(): c for c in t['columns']} for t in schema}
        self.edges = defaultdict(list)
        for t in schema:
            child = t['name'].lower()
            for fk in t['foreign_keys']:
                parent = fk['table'].lower()
                if not fk.get('to') or parent not in self.cols or parent == child:
                    continue
                a, b = fk['from'].lower(), fk['to'].lower()
                self.edges[child].append((parent, a, b))
                self.edges[parent].append((child, b, a))

    def paths(self, starts, end, question):
        """Bounded simple FK paths. Rank short paths by named bridge entities."""
        queue = deque((t, [], {t}) for t in sorted(starts))
        paths = []
        while queue and len(paths) < 24:
            t, path, seen = queue.popleft()
            if t == end and path:
                paths.append(path)
                continue
            if len(path) == 3:
                continue
            for dst, a, b in self.edges[t]:
                if dst not in seen and dst not in starts:
                    queue.append((dst, path+[(t, a, dst, b)], seen | {dst}))
        return sorted(paths, key=lambda p: (len(p), -sum(len(words(e[2]) & question) for e in p), p))

    def owner(self, c, aliases):
        if c.table:
            return c.table.lower() if c.name.lower() in self.cols.get(aliases.get(c.table.lower()), {}) else None
        options = [a for a, t in aliases.items() if c.name.lower() in self.cols.get(t, {})]
        return options[0] if len(options) == 1 else None


def flat(tree):
    return len(list(tree.find_all(exp.Select))) == 1


def add_path(tree, path, aliases):
    out = tree.copy()
    by_name = {t: a for a, t in aliases.items()}
    for src, a, dst, b in path:
        if dst in by_name or dst in aliases:
            return None
        out = out.join(table(dst), on=column(by_name[src], a).eq(column(dst, b)), copy=False)
        by_name[dst] = dst
    return out


def fix_missing(tree, graph, qwords):
    aliases = scope(tree)
    if not aliases or not flat(tree) or not set(aliases.values()) <= graph.cols.keys():
        return
    outputs = {p.alias.lower() for p in tree.expressions if p.alias}
    unresolved = [c for c in tree.find_all(exp.Column) if c.name.lower() not in outputs
                  and not graph.owner(c, aliases)
                  and (c.table or not any(c.name.lower() in graph.cols[t] for t in aliases.values()))]
    if not unresolved:
        return
    first = unresolved[0]
    options = [t for t, cols in graph.cols.items() if t not in aliases.values() and first.name.lower() in cols]
    if first.table.lower() in options:
        options = [first.table.lower()]
    plans = []
    for t in options:
        for path in graph.paths(set(aliases.values()), t, qwords)[:2]:
            plans.append((len(path), -len(words(t) & qwords), path, t))
    for _, _, path, target in sorted(plans)[:3]:
        out = add_path(tree, path, aliases)
        if out is None:
            continue
        # Bind old columns as well: the new endpoint can introduce ambiguities.
        before = list(tree.find_all(exp.Column))
        original = tree.copy()
        valid = True
        for c, old in zip(original.find_all(exp.Column), before):
            a = graph.owner(old, aliases)
            if old in unresolved:
                if old.name.lower() not in graph.cols[target]:
                    valid = False
                    break
                a = target
            if a:
                c.set('table', exp.to_identifier(a, quoted=True))
        if valid:
            original.set('joins', original.args.get('joins', []) + out.args['joins'][len(tree.args.get('joins', [])):])
            yield 'join_missing_fk', original


def prune(tree, graph, qwords):
    aliases = scope(tree)
    joins = tree.args.get('joins', [])
    if not joins or not aliases or not flat(tree) or any(j.side or j.args.get('using') or not isinstance(j.args.get('on'), exp.EQ) for j in joins):
        return
    # SELECT * depends on every table. COUNT(*) depends on multiplicity; here
    # the question must name the entity whose rows are to be counted.
    if any(isinstance(p.unalias(), exp.Star) or p.is_star for p in tree.expressions):
        return
    outer = tree.copy()
    outer.set('joins', [])
    required = set()
    for c in outer.find_all(exp.Column):
        a = graph.owner(c, aliases)
        if a:
            required.add(a)
        elif not any(p.alias.lower() == c.name.lower() for p in tree.expressions if p.alias):
            return
    if not required:
        return
    named = {a for a, t in aliases.items() if words(t) and words(t) <= qwords}
    named.update(a for c in outer.find_all(exp.Column) if words(c.name) & qwords
                 for a in [graph.owner(c, aliases)] if a)
    if not named:
        return
    # Equality endpoints define the actual query graph. Re-root at the first
    # retained relation, so an unnecessary first FROM is removable too.
    edges = []
    for j in joins:
        eq = j.args['on']
        if not all(isinstance(c, exp.Column) for c in (eq.this, eq.expression)):
            return
        u, v = (graph.owner(c, aliases) for c in (eq.this, eq.expression))
        if not u or not v or u == v:
            return
        edges.append((u, v, eq))
    nodes = set(aliases)
    for _ in aliases:
        leaves = {n for n in nodes - (required | named) if sum(n in (u, v) for u, v, _ in edges) == 1}
        if not leaves:
            break
        nodes -= leaves
        edges = [(u, v, eq) for u, v, eq in edges if not leaves.intersection((u, v))]
    if nodes == set(aliases):
        return
    sources = [tree.args['from'].this] + [j.this for j in joins]
    src = {t.alias_or_name.lower(): t for t in sources}
    start = next(a for a in aliases if a in nodes)
    outer = outer.from_(src[start].copy(), copy=False)
    attached = {start}
    while attached != nodes:
        edge = next(((v if u in attached else u, eq) for u, v, eq in edges if (u in attached) != (v in attached)), None)
        if not edge:
            return
        dst, eq = edge
        outer = outer.join(src[dst].copy(), on=eq.copy(), copy=False)
        attached.add(dst)
    yield 'join_prune_leaf', outer


def group_edits(tree, graph, question):
    group = tree.args.get('group')
    if not group or len(group.expressions) != 1 or not scope(tree):
        return
    key = group.expressions[0]
    projected = [p.unalias() for p in tree.expressions]
    if not isinstance(key, exp.Column):
        return
    having = tree.args.get('having')
    if having and all(isinstance(p, exp.Column) for p in projected) and not tree.args.get('limit'):
        # PK -> row attributes is a valid grouped projection. Leave it alone.
        aliases = scope(tree)
        a = graph.owner(key, aliases)
        cols = graph.cols.get(aliases.get(a), {})
        single_pk = sum(bool(c['pk']) for c in cols.values()) == 1 and cols.get(key.name.lower(), {}).get('pk')
        disagrees = any(graph.owner(p, aliases) != a for p in projected)
        if not single_pk or disagrees:
            if not any(p.name.lower() == key.name.lower() and p.table.lower() == key.table.lower() for p in projected):
                inner = tree.select(key.copy(), append=False)
                for field in ('order', 'distinct', 'limit'):
                    inner.set(field, None)
                out = tree.copy()
                out.set('group', None)
                out.set('having', None)
                out = out.where(key.copy().isin(query=inner), copy=False)
                yield 'group_filter_details', out
    if not having and len(projected) == 2 and any(isinstance(p, exp.Count) for p in projected) and re.search(r'\b(each|per|by)\b', question):
        detail = next((p for p in projected if isinstance(p, exp.Column)), None)
        if detail and detail != key and detail.name.lower() != key.name.lower():
            out = tree.group_by(detail.copy(), append=False)
            yield 'group_projected_key', out
            for _, short in prune(out, graph, words(question)):
                yield 'group_prune_leaf', short


def top_edits(tree, graph, question):
    if not tree.args.get('limit') or not tree.args.get('order') or tree.args.get('group') or len(tree.expressions) != 1:
        return
    if not re.search(r'\b(top|bottom|highest|lowest|largest|smallest|first|last|most|least)\b', question):
        return
    p = tree.expressions[0]
    agg = p.unalias()
    if not isinstance(agg, (exp.Sum, exp.Avg, exp.Min, exp.Max, exp.Count)):
        return
    if not isinstance(agg.this, (exp.Column, exp.Star)) or any(not isinstance(o.this, exp.Column) for o in tree.args['order'].expressions):
        return
    counted = exp.Literal.number(1) if isinstance(agg.this, exp.Star) else agg.this.copy()
    inner = tree.select(counted.as_('_value'), append=False)
    inner.set('distinct', None)
    def wrap(inner):
        outagg = agg.copy()
        outagg.set('this', exp.column('_value'))
        return exp.select(outagg.as_(p.alias, quoted=True) if p.alias else outagg).from_(inner.subquery('_ranked'))
    yield 'topk_before_aggregate', wrap(inner)
    # When the measure itself is also the ranking attribute the intended
    # population can be a single relation, without a multiplicity-changing join.
    if isinstance(agg.this, exp.Column):
        alt = inner.copy()
        direction = not re.search(r'\b(bottom|lowest|smallest|least)\b', question)
        alt.set('order', exp.Order(expressions=[exp.Ordered(this=agg.this.copy(), desc=direction, nulls_first=not direction)]))
        for _, short in prune(alt, graph, words(question)):
            yield 'topk_measure_entity', wrap(short)
        yield 'topk_measure', wrap(alt)


NEG = r"\b(without|never|no|not|none|neither)\b|n['’]t\b"


def anti_edits(tree, graph, question):
    if not re.search(NEG, question) or not scope(tree):
        return
    aliases = scope(tree)
    root = tree.args['from'].this
    outer, src = root.alias_or_name.lower(), root.name.lower()
    if src not in graph.cols or not all(all(graph.owner(c, aliases) == outer for c in p.find_all(exp.Column)) for p in tree.expressions):
        return
    where = tree.args.get('where')
    excluded = {e.this.name.lower() for e in tree.find_all(exp.From) if isinstance(e.this, exp.Table) and e.this.name.lower() != src}
    qwords = words(question)
    # A negated membership (or unfiltered LEFT JOIN) is a strong scope cue.
    if not (tree.find(exp.In) or tree.find(exp.Exists) or any(j.side=='LEFT' for j in tree.args.get('joins', [])) or re.search(r'\bwithout\b|\bnever\b|\bany\b', question)):
        return
    if any(tree.args.get(k) for k in ('group','having','limit')):
        return
    # Retain local positive predicates and literal filters inside the negation.
    # Refuse a mixed OR or compound NOT instead of changing its boolean scope.
    keep, filters = [], []
    for term in (where.this.flatten() if where and isinstance(where.this, exp.And) else [where.this] if where else []):
        if term.find(exp.Select):
            if isinstance(term, exp.Or) or term.find(exp.Or):
                return
            for cmp in term.find_all(exp.EQ):
                if isinstance(cmp.this, exp.Column) and isinstance(cmp.expression, exp.Literal):
                    filters.append(cmp)
            # Unsupported inner value filters cannot be silently discarded.
            if any(isinstance(c, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.NEQ, exp.Between)) for c in term.walk()):
                return
            continue
        if isinstance(term, exp.Is) and isinstance(term.expression, exp.Null) and isinstance(term.this, exp.Column) and graph.owner(term.this, aliases) != outer:
            continue
        if all(graph.owner(c, aliases)==outer for c in term.find_all(exp.Column)):
            keep.append(term)
        elif isinstance(term, exp.EQ) and isinstance(term.expression, exp.Literal):
            filters.append(term)
        else:
            return
    paths = []
    for t in graph.cols:
        if t == src:
            continue
        named = words(t) & (qwords - words(src))
        if not named and t not in excluded:
            continue
        for path in graph.paths({src}, t, qwords)[:2]:
            paths.append((-int(bool(named)), len(path), -len(named), path))
        # Some relationship columns lack declared FKs. Require a named target
        # and a PK on the outer relation, plus shared key or a role named in NL.
        if named and not any(n == t for n, _, _ in graph.edges[src]):
            pks = [c for c,m in graph.cols[src].items() if m['pk']]
            if len(pks) == 1:
                for c in graph.cols[t]:
                    role = any(len(w)>=4 and any(v.startswith(w[:4]) for v in qwords) for w in words(c) - {'name','type','date','code'})
                    if c == pks[0] or role and graph.cols[t][c]['type'] == graph.cols[src][pks[0]]['type']:
                        paths.append((0, 1, -len(named), [(src, pks[0], t, c)]))
    for _, _, _, path in sorted(paths)[:5]:
        _, a, dst, b = path[0]
        local = {t: f'_rel{i}' for i, (_, _, t, _) in enumerate(path)}
        if outer in local.values():
            continue
        inner = exp.select('1').from_(table(dst,local[dst])).where(column(local[dst], b).eq(column(outer, a)))
        for prev, a, dst, b in path[1:]:
            inner = inner.join(table(dst,local[dst]), on=column(local[prev], a).eq(column(local[dst], b)), copy=False)
        for cmp in filters:
            c = cmp.this
            owners = [t for t in local if c.name.lower() in graph.cols[t]]
            if len(owners) != 1:
                break
            inner = inner.where(column(local[owners[0]], c.name).eq(cmp.expression.copy()), copy=False)
        else:
            out = tree.copy()
            out.set('joins', None)
            out.set('where', None)
            if out.args.get('order') and any(graph.owner(c,aliases)!=outer for c in out.args['order'].find_all(exp.Column)):
                continue
            for term in keep:
                out = out.where(term.copy(), copy=False)
            yield 'anti_fk_exists', out.where(exp.Not(this=exp.Exists(this=inner)), copy=False)


def correlated_count(tree, graph, question):
    # Include zero-child parents for 'each/per' counts. Move the COUNT into
    # its own correlated SELECT, counting real matches rather than NULL rows.
    joins = tree.args.get('joins', [])
    if len(joins) != 1 or not tree.args.get('group') or not re.search(r'\b(each|every|per|including)\b', question):
        return
    if any(tree.args.get(k) for k in ('where','having','limit','order')) or joins[0].side not in ('', 'LEFT') or not flat(tree):
        return
    aliases = scope(tree)
    if not aliases or not isinstance(joins[0].args.get('on'), exp.EQ):
        return
    count = [p for p in tree.expressions if isinstance(p.unalias(), exp.Count)]
    if len(count) != 1:
        return
    left = tree.args['from'].this
    right = joins[0].this
    for parent, child in ((left,right),(right,left)):
        a, b = parent.alias_or_name.lower(), child.alias_or_name.lower()
        pred = joins[0].args['on'].copy()
        if not all(isinstance(c, exp.Column) and c.table for c in (pred.this,pred.expression)):
            continue
        pk = next((c for c in (pred.this,pred.expression) if c.table.lower()==a), None)
        ck = next((c for c in (pred.this,pred.expression) if c.table.lower()==b), None)
        if not pk or not ck or (aliases[b], pk.name.lower(), ck.name.lower()) not in graph.edges[aliases[a]]:
            continue
        if not graph.cols[aliases[a]][pk.name.lower()]['pk'] or len(tree.args['group'].expressions)!=1:
            continue
        g = tree.args['group'].expressions[0]
        if not isinstance(g, exp.Column) or graph.owner(g, aliases) != a or g.name.lower()!=pk.name.lower():
            continue
        if any(any(graph.owner(c,aliases)!=a for c in p.find_all(exp.Column)) for p in tree.expressions if p is not count[0]):
            continue
        agg = count[0].unalias().copy()
        if any(graph.owner(c,aliases)!=b for c in agg.find_all(exp.Column)):
            continue
        sub = exp.select(agg).from_(child.copy()).where(pred).subquery()
        out = tree.copy().from_(parent.copy(), copy=False)
        out.set('joins',None)
        out.set('group',None)
        out.set('expressions',[sub.as_(p.alias,quoted=True) if p is count[0] and p.alias else sub if p is count[0] else p.copy() for p in tree.expressions])
        yield 'count_correlated', out


def propose(obs, base_sql):
    """Yield distinct (label, SQLite SQL) in priority order, without spending budget."""
    question = obs['task']['question'].lower()
    history = obs.get('history', [])
    schema = next((e['observation'] for e in history if e['tool']=='schema'), [])
    graph = Graph(schema)
    hints = next((e['observation'] for e in history if e['tool']=='repair_hint'), [])
    base = parse(base_sql)
    roots = []
    seen = set()
    for text in [base_sql, *hints[:4]]:
        tree = parse(text)
        if tree is not None and sql(tree) not in seen:
            seen.add(sql(tree))
            roots.append(tree)
    emitted = {sql(base)} if base is not None else set()
    qwords = words(question)
    for idx, tree in enumerate(roots):
        # Restrict hint seeds to the observed projection width.
        if len(tree.expressions) != len(obs['example']['columns']):
            continue
        rules = [top_edits(tree, graph, question), group_edits(tree, graph, question),
                 anti_edits(tree, graph, question), fix_missing(tree, graph, qwords),
                 prune(tree, graph, qwords), correlated_count(tree, graph, question)]
        for stream in rules:
            for label, candidate in stream:
                text = sql(candidate)
                if text not in emitted and len(text.encode()) <= 16000:
                    emitted.add(text)
                    yield label, text
        if idx and base is not None and flat(tree) and set(scope(tree).values()) != set(scope(base).values()):
            # An alternative JOIN spine explicitly names a relation absent in
            # the base and removes intermediates not mentioned in the request.
            new = set(scope(tree).values()) - set(scope(base).values())
            removed = set(scope(base).values()) - set(scope(tree).values())
            if removed and any(words(t) and words(t) <= qwords for t in new) and not any(words(t) and words(t) <= qwords for t in removed):
                if all(isinstance(p.unalias(),exp.Column) for p in tree.expressions) and [p.name.lower() for p in tree.expressions] == [p.name.lower() for p in base.expressions]:
                    # Transplant the relation structure, keeping baseline bag/set semantics.
                    revised = tree.copy()
                    revised.set('distinct', base.args.get('distinct'))
                    text = sql(revised)
                    if text not in emitted:
                        emitted.add(text)
                        yield 'join_named_hint', text


def checked(history, text):
    """Only normalize harmless terminal whitespace; preserve identifier/literal case."""
    key = text.strip().rstrip(';').strip()
    for e in history:
        if e['tool']=='check' and e['sql'].strip().rstrip(';').strip()==key:
            return e['observation']['matches_example']
    return None


class RelationStage:
    """Standalone state for one intercepted action; finish or delegate to parent.

    Feed the next baseline action to intercept(). If it returns None, forward
    the original action. Once it returns an action use advance(obs) until submit.
    """
    def intercept(self, obs, action):
        history = obs['history']
        matched = any(e['tool']=='check' and e['observation']['matches_example'] for e in history)
        self.last_slot = action['tool']=='check' and obs['remaining']==1 and any(e['tool']=='schema' for e in history)
        if action['tool']!='submit' and not self.last_slot:
            return None
        self.base = action['sql']
        base = parse(self.base)
        self.pool = list(propose(obs,self.base))
        # Public agreement alone is weak for empty results. Only replace a
        # passing base if the rewrite also provides strong structural evidence.
        if matched:
            # COUNT(*) is especially sensitive to pruning the counted child.
            # Require an explicitly named population to remain joined, or a
            # DISTINCT argument naming the entity in the question.
            supported = {'join_named_hint','group_filter_details','topk_before_aggregate'}
            qwords = words(obs['task']['question'])
            if base is not None:
                has_distinct_measure = any(c.find(exp.Distinct) and any(words(v.name) & qwords for v in c.find_all(exp.Column)) for c in base.find_all(exp.Count))
                self.pool = [(k,s) for k,s in self.pool if k in supported or
                             (k=='join_prune_leaf' and base.find(exp.Count) and
                              (has_distinct_measure or len(set(scope(parse(s)).values()))>=2 and
                               sum(bool(words(t)) and words(t)<=qwords for t in set(scope(parse(s)).values()))>=2))]
            else:
                self.pool = []
            if self.last_slot:
                # Intercept a passing trajectory only for a removable leaf on
                # the exact pending query, never a speculative hint variant.
                self.pool = [(k,s) for k,s in self.pool if k=='join_prune_leaf' and has_distinct_measure and
                             set(scope(parse(s)).values()) < set(scope(base).values())]
        self.label = None
        self.attempts = []
        return self._next(obs)

    def _next(self, obs):
        for k, s in self.pool:
            known = checked(obs['history'],s)
            if known is True:
                self.label = k
                return {'tool':'submit','sql':s}
            if known is None and obs['remaining']:
                self.attempts.append((k,s))
                return {'tool':'check','sql':s}
        return None

    def advance(self, obs):
        return self._next(obs) or {'tool':'submit','sql':self.base}
