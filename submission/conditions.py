"""Terminal question-to-predicate agreement overlay. Runtime input is one public episode."""
from dataclasses import dataclass
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


OPS = {'=': exp.EQ, '!=': exp.NEQ, '>': exp.GT, '<': exp.LT, '>=': exp.GTE, '<=': exp.LTE, 'like': exp.Like}
WORD_NUM = dict(zip('zero one two three four five six seven eight nine ten single'.split(), '0 1 2 3 4 5 6 7 8 9 10 1'.split()))
NUMBER = r'(?:\d{4}-\d\d-\d\d|-?\d+(?:\.\d+)?|' + '|'.join(WORD_NUM) + r')\b'
PHRASES = {
    '>=': 'no less than|at least', '<=': 'no more than|at most',
    '>': 'greater than|more than|larger than|higher than|bigger than|older than|later than|after|above|over',
    '<': 'less than|fewer than|lower than|smaller than|younger than|earlier than|before|below|under',
    '=': 'equal to|exactly', '!=': 'is not|are not|not equal to',
}
CUES = {p: op for op, phrases in PHRASES.items() for p in phrases.split('|')}
RANGE = re.compile(r'\bbetween\s+('+NUMBER+r')\s+and\s+('+NUMBER+r')', re.I)
COMPARE = re.compile(r'\b('+'|'.join(sorted(CUES, key=len, reverse=True))+r')\s+(?:a\s+)?(?:age\s+)?('+NUMBER+r')', re.I)
QUOTED = r'''(?:"([^"\n]+)"|'([^'\n]+)')'''
MATCHER = re.compile(r'\b(contains?|containing|mentions?|starts? with|begins? with|ends? with|about the topic)\s+(?:(?:the )?(?:word|letter|phrase|substring|character)\s+)?(?:'+QUOTED+r'|([\w-]+))', re.I)
STOP = set('a the and or of that in from for with to by per on is are was were which who whose than had have has all as it their show give find return list what name names of'.split())


def words(text):
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text).casefold()
    def stem(w):
        if w.endswith('ies'): return w[:-3]+'y'
        if len(w)>4 and w.endswith('ed'): return w[:-2]
        if len(w)>3 and w.endswith('s'): return w[:-1]
        return w
    return [stem(w) for w in re.findall('[a-z]+', text)]


@dataclass(frozen=True)
class Constraint:
    family: str
    op: str
    value: str
    context: str
    high: str = ''


def extract(question):
    """Capture the local noun phrase, independently of literals in the SQL."""
    q = question
    def context(m):
        left = re.split(r'\b(?:or|and|whose|where|that|which|who)\b|[,?]', q[:m.start()], flags=re.I)[-1]
        right = re.split(r'\b(?:or|and|ordered|sorted|from|for each)\b|[,?.]', q[m.end():], flags=re.I)[0]
        return ' '.join(left.split()[-3:]+right.split()[:4])
    def number(x): return WORD_NUM.get(x.lower(), x)
    for m in RANGE.finditer(q):
        yield Constraint('range','between',number(m[1]), context(m), number(m[2]))
    for m in COMPARE.finditer(q):
        cue=m[1].lower(); field=context(m)
        if 'older' in cue or 'younger' in cue or 'age ' in m[0].lower(): field='age'
        if cue in ('before','after','earlier than','later than'):
            field += ' date' if '-' in m[2] else ' year'
        op=CUES[cue]
        if re.search(r'\b(?:except|other than) those (?:whose|that)\b[^,?]*$',q[:m.start()],re.I):
            op={'>':'<=','<':'>=','>=':'<','<=':'>'}.get(op,op)
        yield Constraint('date' if cue in ('before','after','earlier than','later than') else 'numeric',op,number(m[2]),field)
    for m in re.finditer(r'\b(?:in (?:the )?(?:year(?: of)? )?|year )(\d{4})\b(?: or (after|before))?',q,re.I):
        yield Constraint('date', {'after':'>=','before':'<='}.get((m[2] or '').lower(),'='), m[1], 'year')
    match_spans=[]
    for m in MATCHER.finditer(q):
        val=next(x for x in m.groups()[1:] if x is not None)
        if '%' in val or '_' in val: continue
        pattern=(val+'%' if m[1].lower().startswith(('start','begin')) else '%'+val if m[1].lower().startswith('end') else '%'+val+'%')
        yield Constraint('matcher','like',pattern,context(m)); match_spans.append(m.span())
    for m in re.finditer(QUOTED,q):
        if any(i<=m.start()<j for i,j in match_spans): continue
        # Quoted target following a field/attribute, excluding aggregate/set scopes.
        left=q[:m.start()]; head=re.split(r'\b(?:and|or|with|whose|under|which|the)\b|[,?]',left,flags=re.I)[-1]
        head=re.sub(r'\b(?:is|are|as|of|not|in|named|called)\b',' ',head,flags=re.I).strip()
        if not head: continue
        neg=bool(re.search(r'\bnot\s*$',left,re.I))
        yield Constraint('domain','!=' if neg else '=',m[1] or m[2],head)
    # Unquoted proper names in an explicit attribute slot, e.g. 'title is ...'.
    for m in re.finditer(r'\b([A-Za-z]+(?:\s+(?:name|code|type))?)\s+(?:is|as|of)\s+([A-Z][\w-]*(?:\s+[A-Z][\w-]*)*)\b',q):
        yield Constraint('domain','=',m[2],m[1])


def parse(sql):
    try: tree=sqlglot.parse_one(sql,read='sqlite')
    except (SqlglotError, RecursionError): return None
    if not isinstance(tree,exp.Select) or tree.find(exp.Subquery, exp.CTE) or any(isinstance(x,exp.Select) and x is not tree for x in tree.walk()): return None
    return tree


def sources(tree, schema):
    catalog={t['name'].casefold():t for t in schema}
    tables=list(tree.find_all(exp.Table))
    if any(t.name.casefold() not in catalog for t in tables): return []
    return [(t.alias_or_name, c['name'], c['type']) for t in tables for c in catalog[t.name.casefold()]['columns']]


def unlower(node):
    return node.this if isinstance(node,exp.Lower) else node


def sql(node): return node.sql(dialect='sqlite')


def norm(text): return ' '.join(str(text).strip().lower().split())


def field_score(c, field):
    q=set(words(c.context)); w=set(words(field)); inter=q&w
    # "name" is often the whole attribute in 'whose name contains ...'.
    if w and w<=q: return 3+len(w)+(3 if list(words(c.context))[-len(w):]==words(field) else 0)
    hit=inter-STOP
    return 1+len(hit)/len(w) if hit else 0


def atoms(tree):
    for clause in ('where','having'):
        node=tree.args.get(clause)
        if node:
            for p in node.find_all(*OPS.values(),exp.Between):
                if p.find_ancestor(exp.Not): continue
                rhs = p.args.get('low') if isinstance(p,exp.Between) else p.expression
                rhs = unlower(rhs)
                if isinstance(rhs,exp.Literal): yield clause,p,rhs.name


def bind(c, tree, cols, allow_insert):
    """Find the predicate slot by independent field evidence or a unique threshold."""
    text=c.family in ('matcher','domain'); value=c.value.strip('%')
    relevant=[]
    for clause,p,raw in atoms(tree):
        left=unlower(p.this); col=left if isinstance(left,exp.Column) else left.find(exp.Column)
        same=norm(raw.strip('%'))==norm(value)
        if (not text and isinstance(p.expression,exp.Literal) and p.expression.is_string and not re.fullmatch(r'-?\d+(?:\.\d+)?|\d{4}-\d\d-\d\d',raw)): continue
        if isinstance(left,exp.AggFunc):
            if text or c.family!='numeric': continue
            strength=4 if same or clause=='having' else 0
        elif col is not None:
            strength=field_score(c,col.name)
        else: continue
        if same or strength>=3: relevant.append((int(same)*5+strength,clause,p,raw))
    relevant.sort(key=lambda x:-x[0])
    if len(relevant)>1 and relevant[0][0]==relevant[1][0]: return None
    slot=relevant[0] if relevant else None
    existing=slot[2] if slot else None
    ranked=[]
    for alias, name, typ in cols:
        if text and re.search(r'INT|REAL|FLOAT|BLOB|DOUBLE',typ,re.I): continue
        if not text and c.family!='date' and re.search(r'CHAR|TEXT|CLOB',typ,re.I): continue
        score=field_score(c,name)
        if score: ranked.append((score,alias,name))
    ranked.sort(reverse=True)
    best=ranked[0] if ranked and (len(ranked)==1 or ranked[0][0]>ranked[1][0]) else None
    if existing is not None:
        left=unlower(existing.this)
        if isinstance(left,exp.Column) and best and best[0]>=3 and best[0]>field_score(c,left.name):
            left=exp.column(best[2],table=best[1],quoted=True)
        else: left=existing.this.copy()
    else:
        if not allow_insert or not best or best[0]<3: return None
        left=exp.column(best[2],table=best[1],quoted=True)
    def lit(value):
        return exp.Literal.string(value) if text or '-' in value[1:] else exp.Literal.number(value)
    if c.op=='between':
        # Preserve an equivalent pair of inclusive bounds.
        if any(isinstance(p,exp.GTE) and raw==c.value and p.this==left for _,p,raw in atoms(tree)) and any(isinstance(p,exp.LTE) and raw==c.high and p.this==left for _,p,raw in atoms(tree)): return None
        target=exp.Between(this=left,low=lit(c.value),high=lit(c.high))
    else:
        rhs=lit(c.value)
        if text and isinstance(left,exp.Lower): rhs=exp.Lower(this=rhs)
        target=OPS[c.op](this=left,expression=rhs)
    if existing is not None:
        same_field=left==existing.this or (isinstance(left,exp.Column) and isinstance(existing.this,exp.Column) and left.name.lower()==existing.this.name.lower() and (not existing.this.table or left.table.lower()==existing.this.table.lower()))
        if same_field and type(target)==type(existing) and norm(slot[3])==norm(c.value) and (c.op!='between' or norm(existing.args['high'].name)==norm(c.high)): return None
        if text and c.family=='domain' and isinstance(existing,exp.Like): return None
        # For an already encoded or more specific domain representation, don't
        # undo value resolution solely because the surface string differs.
        if text and slot[3].casefold()!=c.value.casefold() and c.family=='domain' and (norm(slot[3]) in norm(c.value) or norm(c.value) in norm(slot[3])): return None
        return existing,target,slot[1]
    clause='where'
    if re.search(r'\b(?:or|both|without|except|no|either|not)\b',c.context,re.I): return None
    return None,target,clause


def permitted(q):
    # Set quantifiers and exclusions need scope reasoning outside this stage.
    return not re.search(r'\b(?:both|without|where no|not have any|have no|other than those|except those)\b',q,re.I)


def repairs(text,q,schema,constraints):
    tree=parse(text)
    if tree is None or not permitted(q): return []
    cols=sources(tree,schema)
    if not cols: return []
    # If a grouped frame already contains the count, keep its grouping and
    # aggregate argument. Attach the requested threshold to that aggregate.
    if tree.args.get('group') and not tree.args.get('having') and len(constraints)==1:
        c=constraints[0]; counts=list(tree.find_all(exp.Count))
        if c.family=='numeric' and counts and len({sql(x) for x in counts})==1 and not tree.args.get('where'):
            tree.having(OPS[c.op](this=counts[0].copy(),expression=exp.Literal.number(c.value)),copy=False)
            return [(sql(tree),'having:missing',True)]
    edits=[]
    for c in constraints:
        change=bind(c,tree,cols,not re.search(r'\b(?:or|not)\b',q,re.I) and not tree.find(exp.Or,exp.Not,exp.In))
        if change:
            old,new,clause=change
            if old is None and tree.args.get('where') and any(norm(c.value) in norm(l.name) for l in tree.args['where'].find_all(exp.Literal)): continue
            if old is None and c.family in ('date','numeric','range') and any(isinstance(p,exp.AggFunc) for p in tree.find_all(exp.AggFunc)) and not tree.args.get('where') and tree.args.get('having'): continue
            # Do not introduce a scalar field in place of an implicit group count.
            if old is None and c.family=='numeric' and re.search(r'\b(?:at least|more than|less than) (?:one|two|three|four|a single)\b',q,re.I): continue
            # A compound condition without a corresponding slot/coordination is ambiguous.
            if old is None and len(constraints)>1: continue
            if old is not None: old.replace(new)
            else: tree.where(new,copy=False)
            edits.append(c.family+(':replace' if old is not None else ':missing'))
    if edits: return [(sql(tree),'+'.join(edits),True)]
    return []


def modifier_field(text,q,schema):
    """In 'the X and Y of [modifier] objects', X/Y are output attributes.
    If the draft filters X by the modifier, look for a unique remaining text
    attribute on that object, excluding keys and names. Keep the literal."""
    tree=parse(text)
    if tree is None or tree.args.get('joins') or not tree.args.get('where'): return []
    pred=tree.args['where'].this; tab=tree.find(exp.Table)
    if not tab or not isinstance(pred,exp.EQ) or not isinstance(pred.this,exp.Column) or not isinstance(pred.expression,exp.Literal) or not pred.expression.is_string: return []
    term=pred.expression.name; obj=' '.join(words(tab.name))
    m=re.search(r'\b(?:of|for) (?:all |the )?'+re.escape(term)+r'\s+'+re.escape(obj)+r's?\b',q,re.I)
    if not m or len(tree.expressions)<2: return []
    projected={e.name.lower() for e in tree.expressions if isinstance(e,exp.Column)}
    if pred.this.name.lower() not in projected or not set(words(pred.this.name))<=set(words(q[:m.start()])): return []
    table=next((s for s in schema if s['name'].lower()==tab.name.lower()),None)
    if not table: return []
    fk={k['from'].lower() for k in table.get('foreign_keys',[])}
    fields=[c['name'] for c in table['columns'] if not c['pk'] and c['name'].lower() not in projected|fk and re.search(r'TEXT|CHAR|CLOB',c['type'],re.I) and not set(words(c['name']))&{'id','name','code'}]
    if len(fields)!=1: return []
    pred.set('this',exp.Lower(this=exp.column(fields[0],table=tab.alias_or_name,quoted=True)))
    pred.set('expression',exp.Lower(this=pred.expression.copy()))
    return [(sql(tree),'domain:modifier_field',True)]


def lexical(text,q,schema):
    """Weak branch: literal is grounded but a failed check leaves storage spelling open."""
    tree=parse(text)
    if tree is None or not permitted(q): return []
    cols=sources(tree,schema)
    edited=False
    for _,p,v in list(atoms(tree)):
        if type(p) not in (exp.EQ,exp.NEQ) or not isinstance(p.expression,exp.Literal) or not p.expression.is_string or not isinstance(p.this,exp.Column): continue
        if len(v)<2 or not re.search(r'(?<!\w)'+re.escape(v)+r'(?!\w)',q,re.I): continue
        if not any(n.casefold()==p.this.name.casefold() and (not p.this.table or a.casefold()==p.this.table.casefold()) for a,n,_ in cols): continue
        p.set('this',exp.Lower(this=p.this.copy()));p.set('expression',exp.Lower(this=p.expression.copy()));edited=True
    return [(sql(tree),'domain:case',False)] if edited else []


def agreement(c, tree):
    """Rank literal/operator/field agreement across candidate frames. An absent
    predicate scores zero, even when its literal occurs in SELECT or LIMIT."""
    best=0
    for clause,p,v in atoms(tree):
        if norm(v.strip('%'))!=norm(c.value.strip('%')): continue
        left=unlower(p.this); col=left if isinstance(left,exp.Column) else left.find(exp.Column)
        matched=isinstance(p,exp.Between) and c.op=='between' and p.args['high'].name==c.high
        matched |= (type(p)==OPS.get(c.op) and norm(v)==norm(c.value))
        score=1+3*matched
        if matched and col is not None: score+=min(3,field_score(c,col.name))
        if matched and c.family=='numeric' and clause=='having' and isinstance(left,exp.Count): score+=2
        best=max(best,score)
    return best


def contrasts(text,q,hints,schema,constraints):
    """Require positive agreement on the hint AND a deficit in the submission."""
    parent=parse(text)
    if parent is None or not constraints or not permitted(q): return []
    before=[agreement(c,parent) for c in constraints]; out=[]
    for hint in hints:
        if hint==text: continue
        tree=parse(hint)
        if tree is None or not sources(tree,schema): continue
        after=[agreement(c,tree) for c in constraints]
        if any(a<b for a,b in zip(after,before)): continue
        for c,a,b in zip(constraints,after,before):
            if a>=4 and a>=b+2:
                kind='missing' if not b else 'contradicted'
                out.append((hint,c.family+':hint_'+kind,True)); break
    return out


def options(obs,fallback):
    h=obs['history'];schema=next((e['observation'] for e in h if e['tool']=='schema'),[])
    if not schema: return []
    q=obs['task']['question']; constraints=list(extract(q))
    checked={e['sql']:e['observation']['matches_example'] for e in h if e['tool']=='check'}
    seed_pass=checked.get(fallback,False)
    opts=repairs(fallback,q,schema,constraints)+modifier_field(fallback,q,schema)
    hints=next((e['observation'] for e in h if e['tool']=='repair_hint'),[])
    opts=contrasts(fallback,q,hints,schema,constraints)+opts
    if not seed_pass and not any(checked.values()):
        for t in dict.fromkeys([fallback,*hints]):
            if t!=fallback: opts+=repairs(t,q,schema,constraints)
            opts+=lexical(t,q,schema)
    seen={fallback}; result=[]
    for t,reason,_ in opts:
        if t not in seen and len(t.encode())<=16000 and checked.get(t) is not False:
            result.append((t,reason)); seen.add(t)
    return result


class Overlay:
    """Start at submit only. Keep the exact parent SQL if every check fails."""
    def __init__(self,parent):
        self.parent=parent; self.pending=None; self.reason=None

    def act(self,obs):
        if self.pending is None:
            a=self.parent.act(obs)
            if a['tool']!='submit': return a
            self.fallback=a['sql']; self.pending=options(obs,self.fallback)[:3]
        passed={e['sql']:e['observation']['matches_example'] for e in obs['history'] if e['tool']=='check'}
        for text,reason in self.pending:
            if passed.get(text):
                self.reason=reason; return {'tool':'submit','sql':text}
            if text not in passed and obs['remaining']: return {'tool':'check','sql':text}
        return {'tool':'submit','sql':self.fallback}
