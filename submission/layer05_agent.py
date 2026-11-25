"""Budgeted terminal SQL repairs over the frozen count-aware policy."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from count_agent import Agent as CountAgent, read_json, weights_path
from relations200 import RelationStage
from projection_overlay import Overlay


class Relational:
    def __init__(self, parent):
        self.parent = parent
        self.stage = RelationStage()
        self.active = False

    def act(self, obs):
        if self.active:
            return self.stage.advance(obs)
        pending = self.parent.act(obs)
        action = self.stage.intercept(obs, pending)
        if action is None:
            return pending
        self.active = True
        return action


class Agent(Overlay):
    def __init__(self, cfg, model):
        super().__init__(Relational(CountAgent(cfg, model)))


def create_agent(checkpoint=None):
    return Agent(read_json(checkpoint or HERE/'config.json'),read_json(weights_path())['priority'])
