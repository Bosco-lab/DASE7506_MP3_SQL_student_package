"""Use one optional public SQL probe to resolve remaining passing rivals."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from classic_entity_agent import Agent as EntityAgent, read_json, weights_path
from witness import Witness


class Agent(EntityAgent):
    def __init__(self, config, ranker):
        super().__init__(config, ranker)
        self.witness = None

    def act(self, observation):
        if self.witness is None:
            action = super().act(observation)
            if action['tool'] != 'submit':
                return action
            self.witness = Witness(action['sql'], observation)
        return self.witness.act(observation)


def create_agent(checkpoint=None):
    config = read_json(checkpoint or ROOT / 'config.json')
    return Agent(config, read_json(weights_path())['priority'])
