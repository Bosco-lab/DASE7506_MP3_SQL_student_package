"""Small relational grammar. Identifiers come from schema, predicates from the question.

The measure's source sets the row grain. Bind answer columns separately, then
connect the two with declared keys. Nothing here queries a database or memory.
"""
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


@lru_cache(maxsize=8192)
def words(text):
    def stem(w):
        if w.endswith('ies') and len(w) > 4:
            return w[:-3] + 'y'
        if w.endswith(('sses', 'ches', 'xes')):
            return w[:-2]
        if len(w) > 3 and w.endswith('s') and not w.endswith(('ss', 'us')):
            return w[:-1]
        return w
    text = re.sub('([a-z])([A-Z])', r'\1 \2', text)
    return frozenset(stem(w) for w in re.findall(r'[a-z]+', text.lower()))


def ident(s):
    return '"' + s.replace('"', '""') + '"'


def field(t, c):
    return ident(t) + '.' + ident(c)


@lru_cache(maxsize=1024)
def parse(sql):
    try:
        return sqlglot.parse_one(sql, read='sqlite')
    except (SqlglotError, RecursionError):
        return None


@dataclass(frozen=True)
class Header:
    column: str
    fn: str = ''
    distinct: bool = False

    def sql(self, table):
        col = '*' if self.column == '*' else field(table, self.column)
        return f'{self.fn}({"DISTINCT " if self.distinct else ""}{col})' if self.fn else col


def header(text):
    m = re.fullmatch(r'(?:(count|sum|avg|max|min)\(\s*(distinct\s+)?(.+?)\s*\))', text.strip(), re.I)
    col = m[3] if m else text
    col = col.split('.')[-1].strip('`"[] ')
    if col == '*' and (not m or m[1].upper() != 'COUNT'):
        return None
    return Header(col, m[1].upper() if m else '', bool(m and m[2]))


class Graph:
    def __init__(self, raw):
        self.cols = {t['name']: {c['name'].lower(): c for c in t['columns']} for t in raw}
        canon = {t.lower(): t for t in self.cols}
        self.edges = {t: [] for t in self.cols}
        for t in raw:
            a = t['name']
            for fk in t['foreign_keys']:
                b = canon.get(fk['table'].lower())
                if not b or a == b or not fk['to'] or fk['to'].lower() not in self.cols[b]:
                    continue
                x, y = fk['from'], fk['to']
                self.edges[a].append((a, x, b, y, True))
                self.edges[b].append((b, y, a, x, False))

    def routes(self, src, dest, qwords):
        queue = deque([(src, (), {src})]); found = []
        while queue and len(found) < 24:
            node, path, visited = queue.popleft()
            if node == dest:
                found.append(path)
                continue
            if len(path) >= 3 or (found and len(path) >= len(found[0])):
                continue
            for e in self.edges[node]:
                if e[2] not in visited:
                    queue.append((e[2], path + (e,), visited | {e[2]}))
        def rank(path):
            # Outbound chains preserve fact grain. Repeated roles get lexical support.
            return sum((1 if up else -1.8) + .4*len(words(x) & qwords) for _, x, _, _, up in path)
        return sorted(found, key=rank, reverse=True)


def from_sql(src, path):
    return ' FROM ' + ident(src) + ''.join(' JOIN ' + ident(b) + ' ON ' + field(a, x) + ' = ' + field(b, y)
                                           for a, x, b, y, _ in path)


NUM = {'one': 1, 'single': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10}
NUMBER = r'(?:\d+|' + '|'.join(NUM) + r')'
OPS = {'at least': '>=', 'more than': '>', 'over': '>', 'greater than': '>', 'above': '>', 'bigger than': '>',
       'less than': '<', 'fewer than': '<', 'under': '<', 'at most': '<='}
SUPER = r'\b(most|least|fewest|highest|lowest|maximum|minimum|largest|greatest|smallest|top|latest|earliest|oldest|youngest)\b'
LOW = {'least', 'fewest', 'lowest', 'minimum', 'smallest', 'earliest', 'youngest'}


def number(s):
    return int(s) if s.isdigit() else NUM[s]


def mention(name, bag):
    w = words(name) - {'of', 'in', 'the', 'to', 'ref', 'all'}
    if not w or not w & bag:
        return 0.
    return 2. * len(w & bag) / len(w) + 1.5 * (w <= bag)


@dataclass(frozen=True)
class Candidate:
    score: float
    kind: str
    sql: str


def synthesize(obs, raw):
    q = obs['task']['question'].casefold()
    hs = [header(h) for h in obs['example']['columns']]
    if not hs or None in hs or not any(not h.fn for h in hs):
        return []
    g = Graph(raw)
    qw = words(obs['task']['question'])
    dims = [h for h in hs if not h.fn]
    owners = [t for t, cols in g.cols.items() if all(h.column.lower() in cols for h in dims)]
    if not owners or len({h.column.lower() for h in dims}) != len(dims):
        return []
    # The grouping grammar does not erase value restrictions or negation.
    if re.search(r'''["“”]|(?<!\w)'[^']+'|\b(?:not|never|without|except|between|either|both)\b''', q):
        return []
    threshold = re.search(r'\b(' + '|'.join(OPS) + r')\s+(?:a\s+)?(' + NUMBER + r')\b', q)
    sup = re.search(SUPER, q)
    group_req = any(h.fn for h in hs) or bool(re.search(r'\b(each|per|most common|least common|frequently|frequent)\b', q))
    top = re.search(r'\btop\s+(' + NUMBER + r')\b', q)
    limit = number(top[1]) if top else 1
    direction = 'ASC' if sup and sup[1] in LOW else 'DESC'
    if re.search(r'\bfrom most to least\b', q):
        sup = None
    if not (group_req or sup or threshold or re.search(r'ascending|descending', q)):
        return []
    # Stray numeric/date conditions need their own filter grammar.
    residual = q
    for piece in (threshold, top):
        if piece:
            residual = residual.replace(piece[0], ' ')
    if re.search(r'\d|\b(?:after|before|older than|younger than|where|whose .+ is)\b', residual):
        return []
    ranked = []
    selected_aggs = [h for h in hs if h.fn]
    # Emphasize the noun governed by the counting phrase rather than the group noun.
    metric = re.search(r'\b(?:number of|how many|count of|most|least|fewest)\s+(?:number of\s+)?(?:the\s+)?([\w -]+)', q)
    focus = words(re.split(r'\b(?:for|that|with|do|does|exist|from|are|have|has|in|per|each|which|who|and|across)\b', metric[1])[0]) if metric else set()
    if threshold:
        focus = words(q[threshold.end():].split('.')[0]) - {'order', 'ascending', 'descending'}
    sites = parse(obs['issue_sql'])
    draft_tabs = {t.name.lower() for t in sites.find_all(exp.Table)} if sites else set()
    frequency = bool(re.search(r'\b(common|frequent|frequently|popular|times|count|number|how many|kind)\b', q))
    for dim in owners:
        dim_score = mention(dim, qw) + .4*(dim.lower() in draft_tabs)
        if len(owners) > 1 and not dim_score:
            continue
        dimcols = g.cols[dim]
        for src, cols in g.cols.items():
            if not all(h.column == '*' or h.column.lower() in cols for h in selected_aggs):
                continue
            # A fact source is supported by a relation noun, a named measure/role,
            # or a uniquely relevant incoming relationship for a count per entity.
            srctag = mention(src, qw)
            fact = 1.8*mention(src, focus) + srctag
            attr = max((mention(c['name'], focus) for c in cols.values() if not c['pk']), default=0.)
            fact += .8*attr + .35*(src.lower() in draft_tabs)
            routes = g.routes(src, dim, qw)
            for path in routes[:3]:
                if path and not (srctag or attr or selected_aggs or src.lower() in draft_tabs):
                    continue
                pscore = dim_score + fact - .95*len(path) - 4.5*sum(not e[-1] for e in path)
                if path:
                    pscore += .3*len(qw & words(path[-1][1]))
                proj = ', '.join(h.sql(src if h.fn else dim) for h in hs)
                stem = 'SELECT ' + proj + from_sql(src, path)
                group = [field(dim, h.column) for h in dims]
                key = [c['name'] for c in dimcols.values() if c['pk']]
                # Group named entities by the identifying key carried by the fact;
                # requests for the most common attribute group by its value.
                if path and path[-1][-1] and len(key) == 1 and path[-1][3].lower() == key[0].lower() and mention(dim, qw) and not re.search(r'\b(common|frequent|frequently)\b', q):
                    group = [field(path[-1][0], path[-1][1])]
                elif len(group) > 1 and len(key) == 1 and key[0].lower() in {h.column.lower() for h in dims}:
                    group = [field(dim, key[0])]
                # Numerical measure explicitly near the superlative or total.
                measures = []
                for c in cols.values():
                    cw = words(c['name']) - {'of', 'num', 'number', 'total'}
                    if c['pk'] or not cw or not cw <= qw or 'id' in cw or c['name'].lower() in [h.column.lower() for h in dims]:
                        continue
                    if not re.search(r'int|real|float|double|decimal|numeric', c['type'], re.I):
                        continue
                    measures.append(c['name'])
                if not threshold and (sup or not group_req) and not selected_aggs:
                    ordercols = [c['name'] for c in cols.values() if (not c['pk'] and mention(c['name'], qw) >= 3.5)]
                    if re.search(r'\brecent(?:ly)?\b', q):
                        ordercols += [c['name'] for c in cols.values() if 'date' in words(c['name'])]
                    if not group_req and (not sup or not frequency) and not (top and 'most' in qw) and (not path or srctag):
                        for c in ordercols:
                            if sup and words(c) & {'name','id'}:
                                continue
                            asc = ('DESC' if 'descending' in q else 'ASC') if not sup else direction
                            tail = f' ORDER BY {field(src, c)} {asc}' + (f' LIMIT {limit}' if sup else '')
                            target_words = words(q[sup.end():].split('.')[0]) if sup else qw
                            strength = 2*(c in measures) + 2*mention(c, target_words)
                            ranked.append(Candidate(pscore + 3 + strength, 'rank_attribute', stem + tail))
                if not (selected_aggs or group_req or threshold or (sup and (path or frequency or fact >= 3))):
                    continue
                metric_sql = 'COUNT(*)'
                if measures and re.search(r'\b(total|sum)\b', q):
                    metric_sql = 'SUM(' + field(src, measures[0]) + ')'
                if measures and src == dim and len(key) > 1 and sup and not selected_aggs and not frequency:
                    metric_sql = 'SUM(' + field(src, measures[0]) + ')'
                    pscore += 13  # a measure stored at a finer grain than the requested dimension
                if src == dim and not any(c['name'].lower() in {h.column.lower() for h in dims} for c in cols.values() if c['pk']) and not path:
                    pscore += .6
                if src == dim and len(key) == 1 and (set(words(dim)) <= qw or key[0].lower() in {h.column.lower() for h in dims}):
                    # Counting a table grouped by its own entity gives a constant.
                    pscore -= 6
                if src != dim and path and path[-1][-1] and (frequency or selected_aggs):
                    pscore += 1.5
                tail = ' GROUP BY ' + ', '.join(group)
                if threshold:
                    tail += ' HAVING ' + metric_sql + ' ' + OPS[threshold[1]] + ' ' + str(number(threshold[2]))
                if sup and not (threshold and sup.start() < threshold.end() and sup.end() > threshold.start()):
                    tail += f' ORDER BY {metric_sql} {direction} LIMIT {limit}'
                elif 'descending' in q or 'ascending' in q or 'from most to least' in q:
                    tail += ' ORDER BY ' + metric_sql + (' DESC' if re.search('descending|from most to least', q) else ' ASC')
                elif not (threshold or selected_aggs):
                    continue
                kind = 'threshold' if threshold else 'frequency' if sup else 'group_measure'
                ranked.append(Candidate(pscore + (2 if selected_aggs or frequency or threshold else 0), kind, stem + tail))
    ranked.sort(key=lambda c: -c.score)
    return list(unique(ranked))


def unique(ranked):
    seen = set()
    for c in ranked:
        if c.sql not in seen:
            seen.add(c.sql)
            yield c
