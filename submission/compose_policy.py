"""Budget-aware interruption of the frozen policy. Both live and matched replay
call the same controller. The parent sees its original observations only.
"""
from sqlglot import exp
from compose import parse, synthesize


def checks(obs):
    return {e['sql']: e['observation']['matches_example'] for e in obs['history'] if e['tool'] == 'check'}


def defect(base, proposal):
    tree, alt = parse(base), parse(proposal.sql)
    if not isinstance(tree, exp.Select):
        return True
    if alt is None:
        return False
    if any(alt.args.get(k) and not tree.args.get(k) for k in ('group', 'order', 'having', 'limit')):
        return True
    if len(tree.expressions) != len(alt.expressions):
        return True
    tables = lambda t: {x.name.lower() for x in t.find_all(exp.Table)}
    if tables(alt) < tables(tree) or tree.find(exp.Subquery):
        return True
    if proposal.kind == 'frequency' and tables(alt) != tables(tree) and alt.find(exp.Count):
        return True
    return False


class Stage:
    """The standalone submit-time hook also works with entity_dev.evaluate."""
    def __init__(self, base, obs, *, early=False):
        self.base = base
        raw = next((e['observation'] for e in obs['history'] if e['tool'] == 'schema'), None)
        candidates = synthesize(obs, raw) if raw and obs['remaining'] else []
        # A public check distinguishes plans only on this example. Check the
        # single best grounded plan, never search the pool for accidental fits.
        top = candidates[0] if candidates else None
        supported = top and top.score >= 6 and obs['example']['row_count'] > 0
        self.pool = [(top.kind, top.sql)] if supported and (early or defect(base, top)) and top.sql != base else []

    def act(self, obs):
        done = checks(obs)
        for _, sql in self.pool:
            if done.get(sql):
                return {'tool': 'submit', 'sql': sql}
            if sql not in done and obs['remaining']:
                return {'tool': 'check', 'sql': sql}
        return {'tool': 'submit', 'sql': self.base}


class Controller:
    def __init__(self, parent):
        self.parent = parent
        self.history = []
        self.pending = None
        self.early = None
        self.terminal = None
        self.tried_schema = False
        self.phase = 'parent'
        self.attempts = []
        self.win_source = None

    def act(self, obs):
        # Track which checks were issued by the parent so a failed interruption
        # cannot scramble its schema/hint/model state machine.
        if self.pending:
            event = obs['history'][-1]
            if self.pending == 'parent':
                self.history.append(event)
            elif self.pending == 'early' and event['observation']['matches_example']:
                self.win_source = ('early', self.early.pool[0][0])
                return {'tool': 'submit', 'sql': event['sql']}
            self.pending = None
        if not self.tried_schema and self.history and self.history[-1]['tool'] == 'schema':
            self.tried_schema = True
            if obs['remaining'] >= 2 and not any(checks(obs).values()):
                self.early = Stage(obs['issue_sql'], obs, early=True)
                action = self.early.act(obs)
                if action['tool'] == 'check':
                    self.attempts.append('early')
                    self.pending = 'early'
                    return action
        if self.terminal is None:
            # Missing parent events correspond exactly to extra compose checks.
            # Reuse an actual cached observation if the parent requests one.
            while True:
                logical = dict(obs, history=self.history, remaining=obs['remaining'] + len(obs['history']) - len(self.history))
                action = self.parent.act(logical)
                if action['tool'] == 'submit':
                    break
                cached = next((e for e in obs['history'] if all(e.get(k) == v for k, v in action.items())), None)
                if cached is not None:
                    self.history.append(cached)
                    continue
                if obs['remaining']:
                    self.pending = 'parent'
                    return action
                # An unsuccessful early check can consume the final parent call.
                # Prefer checked successes, then the parent's proposed check.
                passing = [sql for sql, ok in checks(obs).items() if ok]
                sql = passing[-1] if passing else action.get('sql') if action['tool'] == 'check' else obs['issue_sql']
                self.phase = 'shortened'
                return {'tool': 'submit', 'sql': sql}
            self.terminal = Stage(action['sql'], obs)
        action = self.terminal.act(obs)
        if action['tool'] == 'check':
            self.attempts.append('terminal')
            self.pending = 'terminal'
        elif action['sql'] != self.terminal.base:
            self.win_source = ('terminal', self.terminal.pool[0][0])
        return action
