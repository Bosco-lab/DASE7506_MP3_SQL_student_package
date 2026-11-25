"""Budgeted semantic repair and value grounding with independent model proposals."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from parent_agent import Agent as Base, legacy, read_json, weights_path, sql_key
from goal_semantics import semantic_proposals
from value_search import lookup, repaired
from finish import options


class Agent(Base):
    def __init__(self, cfg, model, weights=None):
        super().__init__(cfg, model)
        self.weights = weights if weights is not None else read_json(ROOT / 'distinct.json')
        self.plan = None
        self.did_lookup = False
        self.grounded = []

    def act(self, obs):
        history = obs['history']
        last = history[-1] if history else {}
        if last.get('tool') == 'check' and self.sources.get(last['sql']) in ('guided', 'grounded', 'editor') and last['observation']['matches_example']:
            return {'tool':'submit', 'sql':last['sql']}
        if last.get('tool') == 'schema' and self.schema_info is None:
            self.schema_info = legacy.parse_schema(last['observation'])
        if self.schema_info and self.hints_fetched:
            candidates = [dict(sql=e['sql'], source=self.sources.get(e['sql'], 'hint'), result=e['observation'], order=i)
                          for i,e in enumerate(history) if e['tool']=='check']
            done = {sql_key(c['sql']) for c in candidates}
            passing = [c for c in candidates if c['result']['matches_example']]
            if self.plan and last.get('tool') == 'execute' and last['sql']==self.plan[0]:
                self.grounded = repaired(obs, self.schema_info, self.plan, last['observation']['rows'])
            if obs['remaining']:
                for sql in semantic_proposals(obs, self.schema_info, candidates):
                    if sql_key(sql) not in done and self.worth_check(sql, obs):
                        return self._check(sql, 'guided')
                for sql in self.grounded:
                    if sql_key(sql) not in done and self.worth_check(sql, obs):
                        return self._check(sql, 'grounded')
                if not self.did_lookup and obs['remaining'] >= 2:
                    self.did_lookup = True
                    self.plan = lookup(obs, self.schema_info)
                    if self.plan:
                        return {'tool':'execute','sql':self.plan[0]}
            if passing and self.budget.first_ready:
                # Spend any available spare check on a concrete defect before a model call.
                base = max(passing, key=lambda c: c['source']=='hint')['sql']
                ranked = options(obs, base, self.schema_info['tables'], self.raw_hint_list, self.weights)
                for confidence,kind,sql in ranked:
                    if obs['remaining'] and kind in {'intersection','anti_scope','projection','frequency'} and sql_key(sql) not in done:
                        return self._check(sql,'editor')
        return super().act(obs)


def create_agent(checkpoint=None):
    return Agent(read_json(checkpoint or ROOT/'config.json'), read_json(weights_path())['priority'])
