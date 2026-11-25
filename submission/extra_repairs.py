"""Additional SQLite query repairs using only the public episode and checked queries."""
from __future__ import annotations

from collections import Counter
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


_AGG = {'count': exp.Count, 'sum': exp.Sum, 'avg': exp.Avg, 'max': exp.Max, 'min': exp.Min}


def _parse(sql):
    if not isinstance(sql, str) or not sql or len(sql) > 6000:
        return None
    try:
        nodes = sqlglot.parse(sql, read='sqlite')
        if len(nodes) == 1 and isinstance(nodes[0], (exp.Select, exp.SetOperation)):
            return nodes[0]
    except (SqlglotError, RecursionError):
        pass
    return None


def _sql(node):
    return node.sql(dialect='sqlite', comments=False)


def _key(tree, schema):
    # Ignore redundant qualification and output aliases only in plain SELECTs.
    # Retain aliases when clauses might use them. Never fold literal values.
    def normalize(n):
        if isinstance(n, exp.Identifier):
            n.set('this', n.name.lower())
            if re.fullmatch(r'[a-z_][a-z_0-9]*', n.name):
                n.set('quoted', False)
        return n
    node = tree.transform(normalize)
    if isinstance(node, exp.Select) and not node.find(exp.Subquery) and not node.find(exp.CTE):
        tables = _tables(node)
        owners = {t.alias_or_name: schema.get(t.name, {}).get('columns', {}) for t in tables}
        if tables and all(owners.values()) and len(tables) == 1 + len(node.args.get('joins') or []):
            used_names = {c.name for c in node.find_all(exp.Column) if not c.table}
            for e in list(node.expressions):
                if e.alias and e.alias not in used_names:
                    e.replace(e.unalias().copy())
            for c in node.find_all(exp.Column):
                owner = [t for t, cols in owners.items() if c.name in cols]
                if len(owner) == 1 and c.table == owner[0]:
                    c.set('table', None)
    return _sql(node)


def _tables(tree):
    frm = tree.args.get('from') or tree.args.get('from_')
    sources = ([frm.this] if frm else []) + [j.this for j in tree.args.get('joins') or []]
    return [t for t in sources if isinstance(t, exp.Table)]


def _set(tree, **changes):
    out = tree.copy()
    for k, v in changes.items():
        if isinstance(v, list):
            v = [x.copy() for x in v]
        elif v is not None:
            v = v.copy()
        out.set(k, v)
    return out


def _identifier(name):
    return exp.to_identifier(name, quoted=True)


def _column(name, alias):
    return exp.Column(this=_identifier(name), table=_identifier(alias))


def _headers(headers, sources, schema, original):
    """Bind headers using only visible tables. Retain multiple valid name owners."""
    choices = []
    for i, text in enumerate(headers):
        if not isinstance(text, str):
            return []
        text = text.strip()
        m = re.fullmatch(r'(count|sum|avg|max|min)\s*\(\s*(distinct\s+)?([^()]+?)\s*\)', text, re.I)
        agg, distinct, name = (m.group(1).lower(), bool(m.group(2)), m.group(3)) if m else ('', False, text)
        name = name.rsplit('.', 1)[-1].strip().strip('"`[]').lower()
        hits = []
        for t in sources:
            actual = schema.get(t.name.lower(), {}).get('columns', {}).get(name)
            if actual:
                col = _column(actual, t.alias_or_name)
                if distinct:
                    col = exp.Distinct(expressions=[col])
                hits.append(_AGG[agg](this=col) if agg else col)
        if name == '*' and agg == 'count':
            hits = [exp.Count(this=exp.Star())]
        if not hits:
            return []
        if i < len(original):
            orig = original[i].find(exp.Column)
            if orig and orig.table:
                hits.sort(key=lambda e: not any(c.table.lower() == orig.table.lower() for c in e.find_all(exp.Column)))
        choices.append(hits[:3])
    if not choices:
        return []
    # Primary binding, then one ambiguous slot at a time; avoid Cartesian growth.
    main = [h[0] for h in choices]
    options = [main]
    for i, hits in enumerate(choices):
        for alt in hits[1:]:
            options.append([*main[:i], alt, *main[i+1:]])
    return options[:6]


def _columns_ok(tree, schema):
    """Screen local references; inner selects may legitimately be correlated."""
    tables = _tables(tree)
    frm = tree.args.get('from') or tree.args.get('from_')
    if not tables or not frm or not isinstance(frm.this, exp.Table) or tree.args.get('with') or tree.args.get('with_'):
        return True
    if any(not isinstance(j.this, exp.Table) for j in tree.args.get('joins') or []):
        return True
    by_alias = {t.alias_or_name.lower(): schema.get(t.name.lower(), {}).get('columns', {}) for t in tables}
    aliases = {e.alias.lower() for e in tree.expressions if e.alias}
    for col in tree.find_all(exp.Column):
        if col.find_ancestor(exp.Select) is not tree or col.name == '*':
            continue
        name = col.name.lower()
        if col.table and name not in by_alias.get(col.table.lower(), {}):
            return False
        if not col.table and name not in aliases and not any(name in names for names in by_alias.values()):
            return False
    return True


def _variants(tree, base, schema, fk, question, example, feedback):
    sources = _tables(tree)
    if not sources:
        return
    headers = example.get('columns') or []
    count = example.get('row_count')
    checked_rows = feedback.get('row_count') if feedback.get('ok') else None
    too_many = checked_rows is not None and count is not None and checked_rows > count
    too_few = checked_rows is not None and count is not None and checked_rows < count
    projected = _headers(headers, sources, schema, tree.expressions)
    options = [tree] + [_set(tree, expressions=p) for p in projected]
    where = tree.args.get('where')
    joins = tree.args.get('joins') or []
    group = tree.args.get('group')
    negative = re.search(r"\b(no|not|never|without|except|neither)\b|n't\b", question)
    for i, variant in enumerate(options):
        side = base - .012 * min(i, 3)
        if i:
            yield 'projection', side + .12, variant
        changed = None if variant.args.get('distinct') else exp.Distinct()
        if not all(e.find(exp.AggFunc) is not None for e in variant.expressions) or group:
            boost = too_many if changed is not None else too_few
            yield 'distinct', side + (.22 if boost else .01), _set(variant, distinct=changed)
        if i > 1:
            continue
        if where:
            stripped = _set(variant, where=None)
            yield 'drop_filter', side + (.16 if too_few else .02), stripped
            if tree.args.get('distinct') is None and too_few:
                yield 'drop_filter', side + .01, _set(stripped, distinct=exp.Distinct())
            if isinstance(where.this, exp.And):
                terms = list(where.this.flatten())
                for j in range(min(4, len(terms))):
                    others = [v for k, v in enumerate(terms) if k != j]
                    yield 'drop_filter', side + .035, _set(variant, where=exp.Where(this=exp.and_(*others)))
        if joins:
            for j in range(min(len(joins), 4)):
                yield 'prune_join', side + .065, _set(variant, joins=joins[:j]+joins[j+1:])
            if any(j.side for j in joins):
                inner = [j.copy() for j in joins]
                for j in inner:
                    j.set('side', None)
                yield 'inner_join', side + .06, _set(variant, joins=inner)
        nonscalar = [e.unalias() for e in variant.expressions if e.find(exp.AggFunc) is None]
        if nonscalar and (group or len(nonscalar) != len(variant.expressions)):
            yield 'group', side + .08, _set(variant, group=exp.Group(expressions=nonscalar))
            if group and joins:
                for t in sources:
                    pk = schema.get(t.name.lower(), {}).get('pks') or []
                    if len(pk) == 1:
                        yield 'group', side + .01, _set(variant, group=exp.Group(expressions=[_column(pk[0],t.alias_or_name)]))
        order = variant.args.get('order')
        if nonscalar and (order or group) and re.search(r'\b(most|least|fewest|largest|smallest|greatest|highest|lowest)\b', question):
            desc = not re.search(r'\b(least|fewest|smallest|lowest)\b', question)
            values = [exp.Count(this=exp.Star())]
            if order and isinstance(order.expressions[0].this, exp.Column):
                values.append(exp.Sum(this=order.expressions[0].this.copy()))
            for v in values:
                yield 'group_extreme', side + .105, _set(variant, group=exp.Group(expressions=nonscalar), having=None,
                    order=exp.Order(expressions=[exp.Ordered(this=v, desc=desc, nulls_first=not desc)]),
                    limit=exp.Limit(expression=exp.Literal.number(1)))
        if group and (variant.args.get('having') or order):
            simplified = variant.transform(lambda n: exp.Count(this=exp.Star()) if isinstance(n, exp.Count) else n)
            yield 'group_count', side + .09, simplified
            if nonscalar:
                yield 'group_count', side + .085, _set(simplified, group=exp.Group(expressions=nonscalar))
        if negative and joins and where and all(e.find(exp.AggFunc) is None for e in variant.expressions):
            rhs = variant.copy()
            for j in rhs.args['joins']:
                j.set('side',None)
            pred = rhs.args['where'].this
            if isinstance(pred, exp.NEQ):
                rhs.set('where', exp.Where(this=exp.EQ(this=pred.this.copy(), expression=pred.expression.copy())))
            elif isinstance(pred, exp.Is) and isinstance(pred.expression, exp.Null):
                rhs.set('where', None)
            if not variant.args.get('order') and not variant.args.get('limit'):
                for t in sources:
                    lhs = exp.select(*[e.copy() for e in variant.expressions]).from_(t.copy())
                    if _columns_ok(lhs, schema) and _columns_ok(rhs, schema):
                        yield 'anti_join', side + .17, exp.Except(this=lhs, expression=rhs.copy(), distinct=True)
        if where and where.find(exp.Literal):
            def fold(n):
                if isinstance(n, (exp.EQ, exp.NEQ)) and isinstance(n.this, exp.Column) and isinstance(n.expression, exp.Literal) and n.expression.is_string:
                    return type(n)(this=exp.Lower(this=n.this.copy()), expression=exp.Lower(this=n.expression.copy()))
                return n
            yield 'literal_case', side + (.13 if too_few else .012), _set(variant, where=where.transform(fold))
    # Rebuild from a single owning table, retaining only clauses that can bind.
    # The owner may occur in another hint, in a redundant join, or only in schema.
    active = {t.name.lower(): t for t in sources}
    for name, meta in schema.items():
        t = active.get(name)
        if t is None:
            if not headers or not any(re.search(r'\b'+re.escape(c)+r'\b', str(h), re.I) for h in headers for c in meta['columns']):
                continue
            t = exp.Table(this=_identifier(meta['name']))
        single = _headers(headers, [t], schema, tree.expressions)
        if not single:
            continue
        if len(sources) == 1 and name in active and not joins:
            continue
        def rebind(n):
            if isinstance(n, exp.Column) and n.name.lower() in meta['columns']:
                return _column(meta['columns'][n.name.lower()], t.alias_or_name)
            return n
        if tree.find(exp.Subquery):
            continue
        frm = 'from_' if 'from_' in tree.args else 'from'
        template = _set(tree, expressions=single[0], joins=[], **{frm:exp.From(this=t.copy())}).transform(rebind)
        yield 'single_table', base + .085, template
        if where:
            yield 'single_table', base + .025, _set(template, where=None)
        if where and len(sources) == 1 and name not in active:
            # Same comparison and literal, with the owner corrected via schema.
            yield 'identifier', base + .11, template
    # Correct a missing/miswired FK edge without guessing table or column names.
    known = set()
    for j, t in enumerate(sources):
        for link in fk.get(t.name.lower(), ()):
            target = link['to_table'].lower()
            if not link.get('to_col') or not link.get('from_col'):
                continue
            other = next((v for v in sources if v.name.lower()==target),None)
            if other is not None and j and other.alias_or_name.lower() in known and j-1 < len(joins):
                new = [v.copy() for v in joins]
                new[j-1].set('on', exp.EQ(this=_column(link['from_col'],t.alias_or_name),expression=_column(link['to_col'],other.alias_or_name)))
                yield 'fk_join', base + .03, _set(options[min(1,len(options)-1)], joins=new)
            elif other is None and target in schema and len(sources) < 5:
                new_t = exp.Table(this=_identifier(schema[target]['name']))
                both = _headers(headers, sources+[new_t], schema, tree.expressions)
                if both and (not projected or not _columns_ok(tree, schema)):
                    cond = exp.EQ(this=_column(link['from_col'],t.alias_or_name),expression=_column(link['to_col'],new_t.name))
                    yield 'fk_join', base + .13, _set(tree, expressions=both[0], joins=[*joins,exp.Join(this=new_t,on=cond)])
        known.add(t.alias_or_name.lower())
    for comp in tree.find_all(exp.GT, exp.LT, exp.GTE, exp.LTE):
        sub = comp.expression
        if isinstance(sub, exp.All) and isinstance(sub.this, exp.Select) and len(sub.this.expressions)==1:
            fn = exp.Max if isinstance(comp,(exp.GT,exp.GTE)) else exp.Min
            inner = _set(sub.this, expressions=[fn(this=sub.this.expressions[0].copy())])
            yield 'subquery', base + .21, tree.transform(lambda n: exp.Subquery(this=inner.copy()) if n == sub else n)


def extra_pool(obs, parsed_schema, hints, checked):
    """Return bounded, deterministic, highest-priority-first additional candidates."""
    schema = (parsed_schema or {}).get('tables') or {}
    fk = (parsed_schema or {}).get('fk_graph') or {}
    if not schema:
        return []
    feedback = {}
    for c in checked or ():
        if isinstance(c, dict) and c.get('sql'):
            feedback[c['sql']] = c.get('result') or {}
    question = (obs.get('task', {}).get('question') or obs.get('question') or '').lower()
    example = obs.get('example') or {}
    parsed = {}
    used = set()
    seen_inputs = set()
    seeds = []
    for s in [*(hints or []), obs.get('issue_sql'), *feedback]:
        if not isinstance(s,str) or s in seen_inputs:
            continue
        seen_inputs.add(s)
        tree = _parse(s)
        if tree is None:
            continue
        canon = _key(tree, schema)
        if s in feedback:
            used.add(canon)
        if canon not in parsed:
            parsed[canon] = len(seeds)
            seeds.append([tree, feedback.get(s,{}), s])
        elif s in feedback:
            seeds[parsed[canon]][1] = feedback[s]
    best = {}
    def add(family, priority, node):
        if isinstance(node, exp.Select) and not _columns_ok(node,schema):
            return
        sql = _sql(node)
        key = _key(node, schema)
        if key in used or len(sql.encode('utf8')) > 16000:
            return
        if key not in best or best[key][0] < priority:
            best[key] = (priority, family, sql)
    selected = [(n,res,s) for n,res,s in seeds if isinstance(n,exp.Select) and sum(1 for _ in n.walk()) < 240][:7]
    for index, (node, result, raw) in enumerate(selected):
        base = .66 - .022 * index
        if raw == obs.get('issue_sql'):
            base -= .035
        if result.get('ok') is False:
            base -= .035
        for family, priority, v in _variants(node, base, schema, fk, question, example, result):
            add(family, priority, v)
        # Transfer clauses between hints while mapping aliases by table identity.
        for donor, _, _ in selected[:5]:
            if donor is node:
                continue
            dest = {t.name.lower():t.alias_or_name for t in _tables(node)}
            mapping = {t.alias_or_name.lower():dest[t.name.lower()] for t in _tables(donor) if t.name.lower() in dest}
            def remap(n):
                if isinstance(n, exp.Column) and n.table.lower() in mapping:
                    n.set('table', _identifier(mapping[n.table.lower()]))
                return n
            for keys in [('expressions', 'distinct'),('where',),('group','having','order','limit')]:
                vals = {k:donor.args.get(k) for k in keys}
                if all(v == node.args.get(k) for k,v in vals.items()) or not any(vals.values()):
                    continue
                for k,v in list(vals.items()):
                    if isinstance(v,list):
                        vals[k] = [x.transform(remap) for x in v]
                    elif v is not None:
                        vals[k] = v.transform(remap)
                add('splice', base + .022, _set(node,**vals))
    # Cap redundant families individually as well as the complete list.
    out = []
    counts = Counter()
    for p, f, sql in sorted(best.values(), key=lambda item:-item[0]):
        if counts[f] >= 12:
            continue
        out.append(dict(sql=sql, source='goal_'+f, priority=round(max(0,min(1,p)),4)))
        counts[f] += 1
        if len(out) == 64:
            break
    return out
