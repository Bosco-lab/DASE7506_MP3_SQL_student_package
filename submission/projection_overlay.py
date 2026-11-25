"""Causal hooks for a parent policy. Pass only the current proposed action."""
from sqlglot import exp
from projection200 import propose, _key, _parse, _schema, _words


def _coherent(base, sql, obs):
    """Unique output owner, exact headers, and question evidence for that entity."""
    t, n = _parse(base), _parse(sql)
    if not t or not n or len(t.expressions) != len(n.expressions) or len(n.expressions) < 2:
        return False
    old, new = [e.unalias() for e in t.expressions], [e.unalias() for e in n.expressions]
    if not all(isinstance(e, exp.Column) for e in old + new):
        return False
    owners = {e.table.casefold() for e in new}
    if len(owners) != 1 or len({e.name.casefold() for e in new}) != len(new):
        return False
    if [_key(e.name) for e in new] != [_key(h) for h in obs['example']['columns']]:
        return False
    qualified = next((table for alias, table in _schema(t, obs['history']) if alias.casefold() in owners), None)
    if not qualified:
        return False
    q = set(_words(obs['task']['question']))
    return bool(q & set(_words(qualified['name'])))


def _passed(obs, base, label, sql):
    """Require evidence independent of public row equality on underdetermined checks."""
    if label in ('observed_order', 'header_order', 'sort_column'):
        return True
    if obs['example']['row_count'] != 0:
        return False
    if label == 'display_binding':
        old, new = _parse(base), _parse(sql)
        return any(a.alias and isinstance(a.unalias(), exp.Column) and isinstance(b, exp.Column)
                   and _key(a.alias) == _key(b.name) != _key(a.unalias().name)
                   for a, b in zip(old.expressions, new.expressions))
    if label == 'column_owner' and _coherent(base, sql, obs):
        old = _parse(base)
        return len({e.unalias().table.casefold() for e in old.expressions}) > 1
    return False


def branch(obs, action):
    """Return (fallback SQL, candidate sequence) or None. A branch owns the tail.

    If the last available check follows schema and every passing skeleton has
    conflicting column order, use that check on a reordered passing skeleton.
    Otherwise wait for submit and retain the parent's fallback.
    """
    hist = obs['history']
    if not hist or not obs['remaining']:
        return None
    if action['tool'] == 'submit':
        base = action['sql']
        passed = any(e['tool'] == 'check' and e['sql'] == base and e['observation']['matches_example'] for e in hist)
        pool = [(label, sql) for label, sql in propose(obs, base)
                if not passed or _passed(obs, base, label, sql)]
        if pool:
            return base, pool[:2]
    if hist[-1]['tool'] != 'schema' or action['tool'] == 'submit' or obs['remaining'] != 1:
        return None
    checks = [e for e in hist if e['tool'] == 'check']
    passing = [e for e in checks if e['observation']['matches_example']]
    if passing:
        # All passing checks have to exhibit the same positional conflict.
        for e in passing:
            choice = next(propose(obs, e['sql']), None)
            if not choice or choice[0] != 'header_order':
                return None
        return passing[0]['sql'], [next(propose(obs, passing[0]['sql']))]
    return None


class Tail:
    def __init__(self, fallback, pool):
        self.fallback, self.pool = fallback, pool

    def act(self, obs):
        checked = {e['sql']: e['observation']['matches_example'] for e in obs['history'] if e['tool'] == 'check'}
        for _, sql in self.pool:
            if checked.get(sql):
                return {'tool': 'submit', 'sql': sql}
            if sql not in checked and obs['remaining']:
                return {'tool': 'check', 'sql': sql}
        return {'tool': 'submit', 'sql': self.fallback}


class Overlay:
    def __init__(self, parent):
        self.parent, self.tail = parent, None

    def act(self, obs):
        if self.tail is None:
            action = self.parent.act(obs)
            change = branch(obs, action)
            if change is None:
                return action
            self.tail = Tail(*change)
        return self.tail.act(obs)
