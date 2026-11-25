"""Terminal local AST repairs with the episode's remaining call budget."""
from local_edits import proposals
from evidence import projections
from finish import options
from aggregate import propose as aggregate_edits
from parent_agent import legacy, sql_key

class Stage:
    def __init__(self,base,obs):
        self.base=base
        self.checked={sql_key(e['sql']):e['observation']['matches_example'] for e in obs['history'] if e['tool']=='check'}
        raw=next((e['observation'] for e in obs['history'] if e['tool']=='schema'),None)
        info=legacy.parse_schema(raw) if raw else None
        local=proposals(obs,base,info) if info else []
        self.pool=[item for item in local if item[1]=='name']
        if info:
            self.pool.extend(aggregate_edits(obs,base,info))
            hints=next((e['observation'] for e in obs['history'] if e['tool']=='repair_hint'),[])
            fixes=options(obs,base,info['tables'],hints,{})
            self.pool.extend(item for item in fixes if item[1] in {'projection','intersection','anti_scope','frequency','literal_case'})
            if self.checked.get(sql_key(base)) is False:
                self.pool.extend((11.,'observed_projection',sql) for sql in projections(obs))
            self.pool.sort(key=lambda item:-item[0])
        self.pending=None

    def act(self,obs):
        if self.pending is not None:
            last=obs['history'][-1]
            self.checked[sql_key(self.pending)]=last['observation']['matches_example']
            self.pending=None
        for strength,kind,sql in self.pool:
            key=sql_key(sql)
            if key==sql_key(self.base):continue
            checked=self.checked.get(key)
            if checked is True:return {'tool':'submit','sql':sql}
            if checked is False:continue
            if obs['remaining']:
                self.pending=sql
                return {'tool':'check','sql':sql}
            if strength>=7 and self.checked.get(sql_key(self.base)) is False:
                return {'tool':'submit','sql':sql}
        return {'tool':'submit','sql':self.base}
