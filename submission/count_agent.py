"""Choose the schema-first scheduler for count projections before issuing a tool action."""
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from classic_combo import Agent as Classic
from schema_combo import Agent as SchemaFirst,read_json,weights_path
from rule_mixin import Rules


class ClassicRules(Rules, Classic):
    pass


class CountRules(Rules, SchemaFirst):
    pass


def count_projection(obs):
    return any(re.match(r'count\s*\(',header,re.I) for header in obs['example']['columns'])


class Agent:
    def __init__(self,cfg,model):
        self.cfg,self.model=cfg,model
        self.policy=None

    def act(self,obs):
        if self.policy is None:
            self.policy=(CountRules if count_projection(obs) else ClassicRules)(self.cfg,self.model)
        return self.policy.act(obs)


def create_agent(checkpoint=None):
    return Agent(read_json(checkpoint or ROOT/'config.json'),read_json(weights_path())['priority'])
