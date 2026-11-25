"""Sidecar to an unchanged parent policy. Its first-schema intervention is replayable."""
from sqlglot import exp
from witness import parse
from values_schema import Catalog
from values_conditions import conditions
from values_lookup import plan


class Overlay:
    def __init__(self, parent):
        self.parent=parent
        self.started=False
        self.active=False
        self.probe=None
        self.candidate=None
        self.hidden=set()
        self.kind=''
        self.accepted=False

    def consider(self, obs):
        cat=Catalog(obs); hist=obs['history']
        checks=[e for e in hist if e['tool']=='check']
        passing=any(e['observation']['matches_example'] for e in checks)
        sqls=[e['sql'] for e in reversed(checks)]+[obs['issue_sql']]+[s for e in hist if e['tool']=='repair_hint' for s in e['observation']]
        trees=[x for x in (parse(s) for s in dict.fromkeys(sqls)) if isinstance(x,exp.Select)]
        for k,s in conditions(cat,trees,passing):
            if s and len(s.encode())<=16000:
                self.kind=k; self.candidate=s;return
        if obs['remaining']>=2 and not any(e['tool']=='execute' for e in hist):
            self.probe=plan(cat,trees,passing,obs)
            if self.probe:self.kind=self.probe.kind

    def act(self,obs):
        hist=obs['history'];last=hist[-1] if hist else {}
        if not self.started and last.get('tool')=='schema' and obs['remaining']:
            self.started=True;self.consider(obs)
            self.active=bool(self.probe or self.candidate)
            if self.probe:
                s=self.probe.query()
                if len(s.encode())<=16000:
                    self.hidden.add(len(hist));return {'tool':'execute','sql':s}
                self.active=False
        if self.active and self.probe and last.get('tool')=='execute':
            rows=last['observation']['rows'];options=list(self.probe.finish(rows)) if last['observation']['ok'] else []
            self.candidate=options[0][1] if options else None
        if self.active and self.candidate:
            prior=next((e for e in hist if e['tool']=='check' and e['sql']==self.candidate),None)
            if prior is not None and prior['observation']['matches_example']:
                self.accepted=True;return {'tool':'submit','sql':self.candidate}
            if prior is None and obs['remaining'] and len(self.candidate.encode())<=16000:
                self.hidden.add(len(hist));return {'tool':'check','sql':self.candidate}
        self.active=False
        clean=[e for i,e in enumerate(hist) if i not in self.hidden]
        # Keep the original parent's state and observations in sync. The outer
        # controller still enforces the real eight-call budget on every action.
        parent_obs=dict(obs,history=clean,remaining=obs['remaining']+len(self.hidden))
        a=self.parent.act(parent_obs)
        if a['tool']=='submit':return a
        if obs['remaining'] and not any(all(e.get(k)==v for k,v in a.items()) for e in hist):return a
        checks=[e for e in reversed(clean) if e['tool']=='check']
        best=next((e for e in checks if e['observation']['matches_example']),next(iter(checks),None))
        return {'tool':'submit','sql':best['sql'] if best else obs['issue_sql']}
