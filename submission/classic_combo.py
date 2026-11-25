"""Choose a schema-grounded intervention using the actual episode budget."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from classic_logic_agent import Agent as LogicAgent, early_options, read_json, weights_path
from parent_agent import sql_key
from values_stage import Overlay
from compose_policy import Stage as ComposeStage


class Agent(LogicAgent):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        self.lookup=None
        self.waiting=None
        self.attempted=False
        self.compose_stage=None

    def checked_action(self,obs,sql):
        for e in obs['history']:
            if e['tool']=='check' and sql_key(e['sql'])==sql_key(sql):
                return {'tool':'submit','sql':e['sql']} if e['observation']['matches_example'] else None
        if obs['remaining']:
            self.waiting=sql
            return self._check(sql,'grounded_intervention')
        return None

    def act(self,obs):
        last=obs['history'][-1] if obs['history'] else {}
        if self.waiting and last.get('sql')==self.waiting:
            self.waiting=None
            if last['observation']['matches_example']:return {'tool':'submit','sql':last['sql']}
        if self.lookup and last.get('tool')=='execute':
            probe=self.lookup;self.lookup=None
            if last['observation']['ok']:
                for _,sql in probe.finish(last['observation']['rows']):
                    a=self.checked_action(obs,sql)
                    if a:return a
                    break
        if not self.attempted and last.get('tool')=='schema' and obs['remaining']:
            self.attempted=True
            if not early_options(obs):
                values=Overlay(None);values.consider(obs)
                if values.candidate:
                    a=self.checked_action(obs,values.candidate)
                    if a:return a
                if values.probe and obs['remaining']>=2:
                    sql=values.probe.query()
                    if len(sql.encode())<=16000 and not any(e['tool']=='execute' and e['sql']==sql for e in obs['history']):
                        self.lookup=values.probe
                        return {'tool':'execute','sql':sql}
                if obs['remaining']>=2 and not any(e['observation'].get('matches_example') for e in obs['history'] if e['tool']=='check'):
                    pool=ComposeStage(obs['issue_sql'],obs,early=True).pool
                    if pool:
                        a=self.checked_action(obs,pool[0][1])
                        if a:return a
        if self.compose_stage is None:
            a=super().act(obs)
            if a['tool']!='submit':return a
            self.compose_stage=ComposeStage(a['sql'],obs)
            self.compose_stage.pool=[entry for entry in self.compose_stage.pool if entry[0]!='frequency']
        return self.compose_stage.act(obs)


def create_agent(checkpoint=None):
    return Agent(read_json(checkpoint or ROOT/'config.json'),read_json(weights_path())['priority'])
