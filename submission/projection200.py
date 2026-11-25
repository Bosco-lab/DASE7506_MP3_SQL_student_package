"""Public-observation outer SELECT repairs. Call propose(obs, base_sql) after a
check or schema, preferably on a repaired relational skeleton before submit.
The generator is pure. A caller should check candidates within its own budget.
"""
import itertools
import re
from collections import Counter

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


def _sql(node):
    return node.sql(dialect='sqlite', comments=False)


def _words(s):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', s)
    return [w[:-3] + 'y' if w.endswith('ies') else w[:-2] if w.endswith('uses') else
            w[:-1] if w.endswith('s') and not w.endswith(('ss', 'us')) else w
            for w in re.findall('[a-z0-9]+', s.lower())]


def _key(s):
    return ''.join(_words(s))


def _head(e):
    e = e.unalias()
    return e.name if isinstance(e, exp.Column) else _sql(e)


def _parse(s):
    if not isinstance(s, str) or len(s.encode()) > 16000:
        return None
    try:
        trees = sqlglot.parse(s, read='sqlite')
        if len(trees) == 1 and isinstance(trees[0], exp.Select):
            return trees[0]
    except (SqlglotError, RecursionError):
        pass
    return None


def _schema(tree, history):
    schema = next((e['observation'] for e in history if e['tool'] == 'schema'), [])
    tables = {t['name'].casefold(): t for t in schema}
    frm = tree.args.get('from') or tree.args.get('from_')
    if not frm or tree.args.get('with') or tree.args.get('with_'):
        return []
    sources = [frm.this, *(j.this for j in tree.args.get('joins', []))]
    bound = []
    for t in sources:
        if not isinstance(t, exp.Table) or t.name.casefold() not in tables:
            return []
        bound.append((t.alias_or_name, tables[t.name.casefold()]))
    return bound


def _column(col, alias):
    return exp.column(col, table=alias, quoted=True)


def _feedback(obs, base):
    return next((e['observation'] for e in reversed(obs['history']) if
                 e['tool'] in ('execute', 'check') and e['sql'] == base), {})


def _matches(old, new, sources):
    old = old.unalias()
    if old == new:
        return True
    if isinstance(old, exp.Column) and isinstance(new, exp.Column) and old.name.casefold() == new.name.casefold():
        if old.table:
            return old.table.casefold() == new.table.casefold()
        return sum(any(c['name'].casefold() == old.name.casefold() for c in t['columns']) for _, t in sources) == 1
    return False


def _permutations(old, headers, fb, example, ordered):
    n = len(old)
    if not 2 <= n <= 6 or n != len(headers):
        return
    original = tuple(range(n))
    norms = [_key(h) for h in headers]
    names = [{_key(_head(e)), _key(e.alias_or_name)} for e in old]
    # Counts are interchangeable here only for finding their position. Keep the
    # original aggregate expression/alias intact when moving it.
    for i, e in enumerate(old):
        if isinstance(e.unalias(), exp.AggFunc):
            names[i].add(e.unalias().sql_name().lower())
    targets = [{k, h.split('(')[0].strip().lower()} if '(' in h else {k}
               for h, k in zip(headers, norms)]
    rows, wanted = fb.get('rows', []), example.get('rows', [])
    full = rows and wanted and fb.get('ok') and not fb.get('truncated') and not example.get('truncated') and len(rows) == len(wanted) and all(len(r) == n for r in rows + wanted)
    def agree(p):
        a = [tuple(r[i] for i in p) for r in rows]
        b = list(map(tuple, wanted))
        return a == b if ordered else Counter(a) == Counter(b)
    already = full and agree(original)
    by_header = []
    for perm in itertools.permutations(original):
        if perm == original:
            continue
        aligned = all(names[i] & target for i, target in zip(perm, targets))
        if full and not already and agree(perm):
            yield 'observed_order', [old[i] for i in perm]
        elif aligned and not all(name & target for name, target in zip(names, targets)):
            by_header.append([old[i] for i in perm])
    for expressions in by_header:
        yield 'header_order', expressions


_AGG = {'count': exp.Count, 'min': exp.Min, 'max': exp.Max, 'avg': exp.Avg, 'sum': exp.Sum}


def _choices(header, old, index, sources, question):
    """Bind one slot, using physical column names before aliases and synonyms."""
    m = re.fullmatch(r'(count|min|max|avg|sum)\s*\(\s*(distinct\s+)?([^()]+?)\s*\)', header, re.I)
    raw = (m[3] if m else header).split('.')
    name = raw[-1].strip().strip('"`[]')
    qualifier = raw[-2].strip().strip('"`[]').casefold() if len(raw) == 2 else ''
    if m and name == '*' and m[1].lower() == 'count' and not m[2]:
        return [(exp.Count(this=exp.Star()), '', 100)]
    target = _key(name)
    if not target:
        return []
    display = {'name', 'title', 'label', 'displayname'}
    ident = {'id', 'identifier', 'code'}
    out = []
    for owner, table in sources:
        owner_words = set(_words(table['name']))
        for c in table['columns']:
            colname = c['name']
            words = _words(colname)
            exact = colname.casefold() == name.casefold()
            if exact or _key(colname) == target:
                score = 100 if exact else 94
            elif not m and ((target in display and set(words) & display) or (target in ident and (set(words) & ident or c.get('pk')))):
                score = 55
            else:
                continue
            if qualifier and qualifier in {owner.casefold(), table['name'].casefold()}:
                score += 20
            # Prefer the requested entity, before relative clauses describing restrictions.
            subject = re.split(r'\b(?:that|which have|who|whose|managed by|where|located|with)\b', question.lower())[0]
            score += 4 * bool(owner_words & set(_words(subject)))
            score += 2 * bool(owner_words & set(_words(question)))
            if owner_words == set(_words(name)):
                score += 3
            node = _column(colname, owner)
            if m:
                arg = exp.Distinct(expressions=[node]) if m[2] else node
                node = _AGG[m[1].lower()](this=arg)
            if index < len(old) and _matches(old[index], node, sources):
                node = old[index]
                score += 1
            out.append((node, owner, score))
    # A header may be a calculated output alias without a physical counterpart.
    if not out:
        out = [(e, '', 75) for e in old if _key(e.alias_or_name) == target or _key(_head(e)) == target]
    return sorted(out, key=lambda x: -x[2])[:5]


def _order_column(tree, obs, sources):
    order = tree.args.get('order')
    if not obs['ordered'] or not order or len(order.expressions) != 1:
        return
    term = order.expressions[0]
    if not isinstance(term.this, exp.Column):
        return
    q = obs['task']['question'].lower()
    after = re.search(r'\b(?:sorted|ordered|sort|order)\s+(?:(?:ascending|descending)\s+)?by\s+(.*)', q)
    of = re.search(r'\b(?:ascending|descending|alphabetical)\s+order\s+of\s+([^?.;]+)', q)
    before = re.search(r'(.+?)\s+in\s+(?:ascending|descending|alphabetical)\s+order\s*[.?!]?$', q)
    if after:
        phrase = re.split(r'\b(?:ascending|descending|then|and)\b', after[1])[0]
    elif of:
        phrase = of[1]
    elif before:
        phrase = re.split(r'\band\b', before[1])[-1]
    else:
        return
    tokens = set(_words(phrase))
    if tokens & {'count', 'number', 'sum', 'total', 'average'}:
        return
    options = []
    for alias, t in sources:
        entity = bool(set(_words(t['name'])) & tokens)
        for c in t['columns']:
            attr = set(_words(c['name']))
            if not attr or not attr <= tokens:
                continue
            options.append((len(attr) + 2 * entity, _column(c['name'], alias)))
    options.sort(key=lambda x: -x[0])
    if options and (len(options) == 1 or options[0][0] > options[1][0]) and not _matches(term.this, options[0][1], sources):
        if not term.this.table and any(e.alias.casefold() == term.this.name.casefold() for e in tree.expressions):
            return
        edit = tree.copy()
        edit.args['order'].expressions[0].set('this', options[0][1])
        yield 'sort_column', edit


def _project(tree, expressions):
    # Preserve dependencies of other clauses. Ordinals cannot silently change
    # meaning, nor can moving/rebinding an alias change HAVING or ORDER BY.
    original = tree.expressions
    other = [e for k in ('where', 'group', 'having', 'order', 'joins')
             for v in [tree.args.get(k)] for e in (v if isinstance(v, list) else [v]) if e]
    references = {c.name.casefold() for e in other for c in e.find_all(exp.Column) if not c.table}
    for e in original:
        if e.alias and e.alias.casefold() in references and e not in expressions:
            return None
    out = tree.copy()
    for key in ('order', 'group'):
        clause = out.args.get(key)
        if clause:
            for item in clause.expressions:
                term = item.this if isinstance(item, exp.Ordered) else item
                if isinstance(term, exp.Literal) and term.is_int:
                    i = int(term.this) - 1
                    if not 0 <= i < len(original) or original[i] not in expressions:
                        return None
                    term.replace(exp.Literal.number(expressions.index(original[i]) + 1))
    out.set('expressions', [e.copy() for e in expressions])
    return out


def propose(obs, base_sql):
    """Yield deterministic, priority-ordered (label, SQL) using only public input."""
    tree = _parse(base_sql)
    headers = obs['example']['columns']
    if tree is None or not 1 <= len(headers) <= 8 or not all(isinstance(h, str) for h in headers):
        return
    old = tree.expressions
    sources = _schema(tree, obs['history'])
    fb = _feedback(obs, base_sql)
    seen = {_sql(tree)}

    def emit(label, node):
        if node is not None:
            text = _sql(node)
            if text not in seen and len(text.encode()) <= 16000:
                seen.add(text)
                yield label, text

    for label, es in _permutations(old, headers, fb, obs['example'], obs['ordered']):
        yield from emit(label, _project(tree, es))
    for label, node in _order_column(tree, obs, sources):
        yield from emit(label, node)
    choices = [_choices(h, old, i, sources, obs['task']['question']) for i, h in enumerate(headers)]
    if not all(choices):
        return
    # Small beam with a bonus for attributes of the same entity (id/name,
    # name/measure, ...). Never enumerate the entire Cartesian product.
    beam = [(0, [], [])]
    for slot in choices:
        ranked = []
        for score, es, owners in beam:
            for e, owner, strength in slot:
                ranked.append((score + strength + 3 * bool(owner and owner in owners), es + [e], owners + [owner]))
        beam = sorted(ranked, key=lambda x: -x[0])[:12]
    for _, es, owners in beam:
        diffs = [i for i, e in enumerate(es) if i >= len(old) or not _matches(old[i], e.unalias(), sources)]
        if not diffs and len(es) == len(old):
            continue
        if len(es) != len(old):
            label = 'projection_arity'
        elif all(isinstance(es[i].unalias(), exp.Column) and isinstance(old[i].unalias(), exp.Column) and es[i].unalias().name.casefold() == old[i].unalias().name.casefold() for i in diffs):
            label = 'column_owner'
        elif any(old[i].alias and _key(old[i].alias) == _key(headers[i]) for i in diffs):
            label = 'display_binding'
        else:
            label = 'public_header' if len(diffs) == 1 else 'multiple_slots'
        yield from emit(label, _project(tree, es))
