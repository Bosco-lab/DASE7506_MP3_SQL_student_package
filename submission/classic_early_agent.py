"""Continue the frozen fusion policy with budgeted terminal repairs."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from classic_fusion_agent import Agent as FusionAgent, read_json, weights_path
from stage import Stage
from classic_early import first_check


class Agent(FusionAgent):
    def __init__(self, config, ranker):
        super().__init__(config, ranker)
        self.terminal_stage = None
        self.early_sql = None

    def act(self, observation):
        events = observation['history']
        if self.early_sql and events[-1].get('sql') == self.early_sql and events[-1]['observation'].get('matches_example'):
            return {'tool': 'submit', 'sql': self.early_sql}
        sql = first_check(observation)
        if sql:
            self.early_sql = sql
            return self._check(sql, 'early_schema')
        if self.terminal_stage is None:
            action = super().act(observation)
            if action['tool'] != 'submit':
                return action
            self.terminal_stage = Stage(action['sql'], observation)
        return self.terminal_stage.act(observation)


def create_agent(checkpoint=None):
    config = read_json(checkpoint or ROOT / 'config.json')
    return Agent(config, read_json(weights_path())['priority'])
