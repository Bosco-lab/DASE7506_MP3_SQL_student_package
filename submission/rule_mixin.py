from semantic200 import options

class Rules:
    def act(self, obs):
        last=obs['history'][-1] if obs['history'] else {}
        if last.get('tool')=='schema':
            self.semantic_pool=list(dict.fromkeys(sql for _,sql in options(obs) if len(sql.encode())<=16000))[:2]
        if getattr(self,'semantic_pool',None) and obs['remaining']:
            if last.get('tool')=='check' and self.waiting==last.get('sql') and last['observation']['matches_example']:
                return {'tool':'submit','sql':last['sql']}
            while self.semantic_pool:
                action=self.checked_action(obs,self.semantic_pool.pop(0))
                if action:return action
        return super().act(obs)


