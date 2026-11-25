"""Condition scope rules. All identifiers and code literals come from the episode."""
import re
from sqlglot import exp
from values_schema import affinity, lit, qi, ref, words


def conditions(cat, trees, passing):
    q=cat.q.lower(); h=cat.headers
    quoted=re.findall(r'''["']([^"']+)["']''',cat.q)
    for tree in trees:
        where=tree.args.get('where')
        if not where: continue
        # A Boolean describing an outcome is a different property from participation.
        # Accept only if a requested role is already encoded by the join itself.
        if isinstance(where.this,exp.EQ) and isinstance(where.this.expression,exp.Boolean):
            role=[c for j in tree.args.get('joins',[]) for c in j.find_all(exp.Column) if len(c.name)>3 and c.name.lower() in q]
            for t,c in cat.resolve(where.this.this,tree):
                unasked=set(words(c)) - cat.qw - {'if','is'}
                if affinity(cat.cols[t][c]['type'])=='bool' and role and unasked and re.search(r'\b(side|role|position)\b',q):
                    fixed=tree.copy();fixed.set('where',None);yield 'role_boolean',fixed.sql(dialect='sqlite')
        # A token already used as part of a named role in a join was spuriously
        # split off into a categorical filter on a different column.
        pred=where.this
        if isinstance(pred,exp.EQ) and isinstance(pred.expression,exp.Literal) and pred.expression.is_string and tree.args.get('joins'):
            token=pred.expression.name.lower(); col=pred.this
            if isinstance(col,exp.Column) and not set(words(col.name)) & cat.qw:
                for j in tree.args['joins']:
                    for key in j.find_all(exp.Column):
                        w=words(key.name)
                        if len(w)>=2 and token in w and set(w)<=cat.qw:
                            fixed=tree.copy();fixed.set('where',None);yield 'role_token',fixed.sql(dialect='sqlite')
        # A local range can be answered at its native table grain.
        for b in where.find_all(exp.Between):
            n,m=b.args['low'],b.args['high']
            if not all(isinstance(x,exp.Literal) and not x.is_string for x in (m,n)): continue
            if not re.search(r'\bbetween\s+'+re.escape(n.name)+r'\s+and\s+'+re.escape(m.name)+r'\b',q): continue
            if passing or len(re.findall(r'\d+(?:\.\d+)?',q))!=2: continue
            for t,c in cat.resolve(b.this,tree):
                base=cat.select([(t,c)],anchor=t,distinct='distinct' in cat.qw)
                if base and affinity(cat.cols[t][c]['type'])=='number':
                    yield 'local_range',base[0]+' WHERE '+ref(t,c)+f' BETWEEN {n.name} AND {m.name}'
        # Either A or (B with C). A global AND would discard all A without C.
        if 'either' in q and ' or ' in q and ' but ' in q:
            isin=where.find(exp.In); eq=where.find(exp.EQ); nums=re.findall(r'\b\d+\b',q)
            if isin and eq and len(nums)==3 and {x.name for x in isin.expressions}==set(nums[:2]) and eq.expression.name==nums[2]:
                fixed=tree.copy();fixed.set('where',None);fixed.set('distinct',None)
                base=fixed.sql(dialect='sqlite')+' WHERE '; col=isin.this.sql(dialect='sqlite')
                yield 'scoped_disjunction',base+col+' = '+nums[0]+' UNION '+base+col+' = '+nums[1]+' AND '+eq.sql(dialect='sqlite')
        # Count distinct entities with some A but no B across multiple records.
        if ' but not ' in q and len(tree.expressions)==1 and tree.find(exp.Count):
            counted=tree.find(exp.Count).this; terms=list(where.find_all(exp.EQ,exp.NEQ))
            if isinstance(counted,exp.Distinct) and len(terms)==2 and {type(x) for x in terms}=={exp.EQ,exp.NEQ} and all(isinstance(x.expression,exp.Literal) for x in terms) and terms[0].this==terms[1].this:
                fixed=tree.copy();fixed.set('where',None);fixed.set('expressions',[counted.expressions[0].copy()]);base=fixed.sql(dialect='sqlite')+' WHERE '
                def select(p):return base+'LOWER('+p.this.sql(dialect='sqlite')+') = '+lit(p.expression.name.lower())
                pos,neg=sorted(terms,key=lambda x:isinstance(x,exp.NEQ));yield 'count_difference','SELECT COUNT(*) FROM ('+select(pos)+' EXCEPT '+select(neg)+')'
        # Quantification over a comparison needs an extremum, whereas SQLite
        # otherwise takes the first row of a scalar subquery.
        if re.search(r'\bthan (?:the \w+ of )?(?:all|every)\b',q):
            for comp in where.find_all(exp.GT,exp.LT):
                sub=comp.expression
                if not isinstance(sub,exp.Subquery) or not isinstance(sub.this,exp.Select):continue
                inner=sub.this
                if len(inner.expressions)!=1 or not isinstance(inner.expressions[0],exp.Column) or not inner.args.get('where') or inner.find(exp.Subquery):continue
                if not isinstance(comp.this,exp.Column) or inner.expressions[0].name.lower()!=comp.this.name.lower():continue
                local=inner.copy();local.set('expressions',[(exp.Max if isinstance(comp,exp.GT) else exp.Min)(this=inner.expressions[0].copy())])
                if local.args.get('from') and isinstance(local.args['from'].this,exp.Table):
                    for c in local.find_all(exp.Column):c.set('table',None)
                    fixed=tree.transform(lambda n:exp.Subquery(this=local) if n==sub else n);yield 'quantified_range',fixed.sql(dialect='sqlite')
    # Bind 'both' to one explicitly named category, at entity grain.
    if len(quoted)==2 and 'both' in cat.qw:
        for t,cols in cat.cols.items():
            for c in cols:
                if not re.search(r'\b'+r'\s+'.join(map(re.escape,words(c)))+r'\s+(?:are\s+)?both\b',q) and c.replace('_',' ').lower()+' both' not in q:continue
                b=cat.select([(t,c)]);
                if b: yield 'category_intersection',' INTERSECT '.join(b[0]+' WHERE '+ref(t,c)+' = '+lit(v) for v in quoted)
    # Entity anti-predicates: two scoped queries, with the same projected columns.
    anti=re.search(r'\b(?:where no|without an?|does not have any|do not have any)\b(.+)',q)
    if anti:
        tail=anti[1]; nums=re.search(r'\b(more than|less than|over|above|under|below)\s+(\d+(?:\.\d+)?)\b',tail)
        tw=set(words(tail)); opts=[]
        for t,cols in cat.cols.items():
            for c in cols:
                cw=set(words(c)); hit=cw & tw
                if nums and hit and not 'id' in cw and not cat.cols[t][c]['pk']:
                    test=ref(t,c)+(' < ' if nums[1] in ('less than','under','below') else ' > ')+nums[2]; strength=len(hit)
                elif len(quoted)==1 and hit and affinity(cols[c]['type'])=='text':
                    test=ref(t,c)+' = '+lit(quoted[0]);strength=len(hit)
                else:continue
                # Resolve the entity before the exclusion; the bridge can have
                # the same header but lacks entities with no children.
                head=set(words(q[:anti.start()]));roots=[u for u in cat.cols if set(words(u))<=head and all(cat.owner(u,hdr) for hdr in h)]
                root=max(roots,key=lambda u:len(words(u)),default=None)
                b=cat.select([(t,c)],anchor=root);
                if not b: continue
                start=cat.select([],anchor=b[1]);
                if not start or (nums and b[1]!=t):continue
                opts.append((strength+len(set(words(t))&tw)*0.25,b[0],start[0],test))
        if opts:
            _,base,start,test=max(opts,key=lambda x:x[0]);yield 'condition_difference',start+' EXCEPT '+base+' WHERE '+test
    # In 'the <entity> called X', bind X to that entity's name. A join
    # introduced just to search the name of a different entity is extraneous.
    named=re.search(r'''\b(\w+)\s+(?:called|named)\s+["']''',q)
    if named and len(quoted)==1 and not re.search(r'\b(?:order|sort|each|per)\b',q):
        entity=words(named[1])[0]
        for tree in trees:
            wh=tree.args.get('where')
            if not wh or not isinstance(wh.this,exp.EQ) or not isinstance(wh.this.expression,exp.Literal) or wh.this.expression.name.lower()!=quoted[0].lower():continue
            for t,c in cat.resolve(wh.this.this,tree):
                if entity not in words(t) or 'name' not in words(c):continue
                if len(tree.expressions)!=len(h):continue
                owners=[cat.resolve(x.unalias(),tree) for x in tree.expressions]
                if not all(len(x)==1 for x in owners) or len({x[0][0] for x in owners})!=1:continue
                b=cat.select([(t,c)],anchor=owners[0][0][0]);
                if b and b[0].count(' JOIN ')<len(tree.args.get('joins') or []):yield 'named_entity',b[0]+' WHERE '+ref(t,c)+' = '+lit(quoted[0])
    # A topic/substring question names the projection as the field to search.
    if len(quoted)==1 and len(h)==1 and re.search(r'\b(about the topic|mentioning|containing the phrase)\b',q):
        for t in cat.cols:
            field=cat.owner(t,h[0]);b=cat.select([],anchor=t)
            if b and field and affinity(cat.cols[t][field[1]]['type'])=='text': yield 'topic_contains',b[0]+' WHERE '+ref(*field)+' LIKE '+lit('%'+quoted[0]+'%')
    # Competing branches (aggregate or matching rows) must preserve both projections.
    if len(quoted)==1 and len(h)==1 and ' or ' in q and 'char' in q:
        m=re.fullmatch(r'(max|min)\(([^()]+)\)',h[0],re.I)
        if m:
            for t in cat.cols:
                field=cat.owner(t,m[2]);
                if field:
                    col=ref(*field);src=' FROM '+qi(t);yield 'aggregate_disjunction',f'SELECT {m[1]}({col})'+src+' UNION SELECT '+col+src+' WHERE '+col+' LIKE '+lit('%'+quoted[0]+'%')
