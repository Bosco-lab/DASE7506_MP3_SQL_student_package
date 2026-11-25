"""Ground the public aggregate output signature on competing relational inputs."""
import re
from sqlglot import exp
from projection200 import _parse, _schema, _sql, _project, _column, _words

_PATTERN = re.compile(r'(count|sum|avg|min|max)\s*\(\s*(distinct\s+)?(?:[\w]+\.)?([\w]+|\*)\s*\)', re.I)
_FUNCS = {'count': exp.Count, 'sum': exp.Sum, 'avg': exp.Avg, 'min': exp.Min, 'max': exp.Max}


def signature(strings):
    slots = []
    for s in strings:
        match = _PATTERN.fullmatch(s.strip().replace('"', '').replace('`', ''))
        if match:
            fn, distinct, col = match.groups()
            slots.append((fn.lower(), bool(distinct), col.lower()))
        elif re.fullmatch(r'\w+', s):
            slots.append(('', False, s.lower()))
        else:
            return None
    return tuple(slots)


def actual(tree):
    if tree is None:
        return None
    return signature([item.name if isinstance(item, exp.Column) else _sql(item)
                      for item in (e.unalias() for e in tree.expressions)])


def bind(tree, slots, obs):
    src = _schema(tree, obs['history'])
    if not src:
        return None
    bound = []
    for fn, distinct, col in slots:
        if col == '*':
            arg = exp.Star()
        else:
            possible = [_column(c['name'], alias) for alias, t in src for c in t['columns'] if c['name'].lower() == col]
            if len(possible) != 1:
                keys = tree.args.get('group')
                cols = list(keys.find_all(exp.Column)) if keys else []
                possible = [x for x in possible if any(x.name.lower() == c.name.lower() and x.table.lower() == c.table.lower() for c in cols)]
            if len(possible) != 1:
                return None
            arg = possible[0]
        if distinct:
            arg = exp.Distinct(expressions=[arg])
        bound.append(_FUNCS[fn](this=arg) if fn else arg)
    return _project(tree, bound)


def options(obs, pending):
    slots = signature(obs['example']['columns'])
    if not slots or not any(s[0] for s in slots):
        return []
    hist = obs['history']
    checks = [e for e in hist if e['tool'] == 'check']
    schema = next((e['observation'] for e in hist if e['tool'] == 'schema'), None)
    if not schema or not checks:
        return []
    passed = [e for e in checks if e['observation']['matches_example']]
    if any(actual(_parse(e['sql'])) == slots for e in passed):
        return []
    if pending['tool'] == 'submit' and actual(_parse(pending['sql'])) == slots and passed:
        return []
    hints = next((e['observation'] for e in hist if e['tool'] == 'repair_hint'), [])
    prior = {e['sql']: e['observation'] for e in checks}
    opts = {}
    # Compound queries take output names from their first branch only.
    if _parse(pending['sql']) is None or any(_parse(e['sql']) is None for e in passed):
        return []
    all_keys = {_sql(t) for e in checks if (t := _parse(e['sql'])) is not None}
    def add(tree, score):
        if tree is not None:
            sql = _sql(tree)
            if sql not in all_keys and len(sql.encode()) <= 16000:
                opts[sql] = max(score, opts.get(sql, -100))
    for i, text in enumerate(dict.fromkeys([*hints, *(e['sql'] for e in reversed(checks))])):
        t = _parse(text)
        if t is None or actual(t) == slots:
            continue
        new = bind(t, slots, obs)
        if new is None:
            continue
        # Keep row semantics: the seed already has grouping for mixed projections.
        if any(not fn for fn, _, _ in slots) and not t.args.get('group'):
            continue
        feedback = prior.get(text, {})
        support = 3 * feedback.get('matches_example', False) + int(feedback.get('ok', False))
        add(new, support + 2 * (text in hints) + .5 * (len(t.expressions) == len(slots)) - .01*i)
    if not passed and all(fn and col != '*' for fn, _, col in slots):
        # A single native scalar aggregate has one output row regardless of an
        # outer ORDER BY/LIMIT (provided the requested limit is positive).
        owners = [t for t in schema if all(any(c['name'].lower() == col for c in t['columns']) for _, _, col in slots)]
        if len(owners) == 1:
            table = owners[0]['name']
            tree = exp.select('*').from_(exp.to_table(table, quoted=True))
            native = bind(tree, slots, obs)
            # Prefer the native grain when the question names its table/metric
            # and there is evidence the scalar has been computed over a join.
            q = set(_words(obs['task']['question']))
            sources = [_parse(e['sql']) for e in checks if e['observation']['ok']]
            plausible = any(t is not None and t.args.get('joins') and not t.args.get('where') for t in sources)
            add(native, 7 if plausible and q & set(_words(table)) else -1)
    return sorted(opts, key=opts.get, reverse=True)


class Overlay:
    def __init__(self, parent):
        self.parent = parent
        self.tail = None

    def act(self, obs):
        if self.tail is None:
            action = self.parent.act(obs)
            remaining = obs['remaining']
            if not remaining or (action['tool'] != 'submit' and (remaining != 1 or action['tool'] != 'check')):
                return action
            pool = options(obs, action)
            if not pool:
                return action
            passed = [e['sql'] for e in obs['history'] if e['tool'] == 'check' and e['observation']['matches_example']]
            fallback = action['sql'] if action['tool'] == 'submit' or not passed else passed[0]
            self.tail = (fallback, pool[:2])
        fallback, pool = self.tail
        results = {e['sql']: e['observation']['matches_example'] for e in obs['history'] if e['tool'] == 'check'}
        for sql in pool:
            if results.get(sql):
                return {'tool': 'submit', 'sql': sql}
            if sql not in results and obs['remaining']:
                return {'tool': 'check', 'sql': sql}
        return {'tool': 'submit', 'sql': fallback}
