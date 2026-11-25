"""Apply entity-grain repairs after the early schema-check agent finishes."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from early_agent import Agent as EarlyAgent, read_json, weights_path
from entity import EntityStage


class Agent(EarlyAgent):
    def __init__(self, config, ranker):
        super().__init__(config, ranker)
        self.entity_stage = None

    def act(self, observation):
        if self.entity_stage is None:
            action = super().act(observation)
            if action['tool'] != 'submit':
                return action
            self.entity_stage = EntityStage(action['sql'], observation)
        return self.entity_stage.act(observation)


def create_agent(checkpoint=None):
    config = read_json(checkpoint or ROOT / 'config.json')
    return Agent(config, read_json(weights_path())['priority'])
