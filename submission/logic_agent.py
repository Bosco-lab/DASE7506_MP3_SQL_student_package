"""Schema-time quantified repairs, followed by a bounded terminal projection check."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from witness_agent import Agent as WitnessAgent,read_json,weights_path
from quantified198 import hypotheses
from ground198 import Stage
from parent_agent import sql_key


def early_options(obs):
    if not obs['remaining'] or not obs['history'] or obs['history'][-1]['tool'] != 'schema':
        return []
    from explicit198 import explicit_options
    from parent_agent import legacy
    info = legacy.parse_schema(obs['history'][-1]['observation'])
    return hypotheses(obs) + explicit_options(obs,info)


class Agent(WitnessAgent):
    def __init__(self, cfg, model):
        super().__init__(cfg,model)
        self.early_attempt=None
        self.terminal=None

    def act(self,obs):
        last=obs['history'][-1] if obs['history'] else {}
        if self.early_attempt and last.get('sql')==self.early_attempt and last['observation'].get('matches_example'):
            return {'tool':'submit','sql':self.early_attempt}
        options=early_options(obs)
        if options:
            checked={sql_key(e['sql']): e['observation']['matches_example'] for e in obs['history'] if e['tool']=='check'}
            for kind,sql in options:
                if checked.get(sql_key(sql)) is True:return {'tool':'submit','sql':sql}
                if sql_key(sql) not in checked:
                    self.early_attempt=sql
                    return self._check(sql,kind)
        if self.terminal is None:
            action=super().act(obs)
            if action['tool']!='submit':return action
            self.terminal=Stage(action['sql'],obs)
        return self.terminal.act(obs)


def create_agent(checkpoint=None):
    return Agent(read_json(checkpoint or ROOT/'config.json'),read_json(weights_path())['priority'])
