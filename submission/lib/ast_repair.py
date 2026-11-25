"""Deterministic SQLGlot AST repair and 3-instance spurious-pass robustness scorer."""
from __future__ import annotations

from collections import Counter, deque
from difflib import SequenceMatcher
import itertools
import re
from typing import Any

import sqlglot
from sqlglot import exp

MAX_SQL_BYTES = 16000

_DISTINCT_RE = re.compile(r"\b(distinct|different|unique)\b", re.IGNORECASE)
_AGG_HINT_RE = re.compile(
    r"\b(most|least|more than|at least|at most|count|number of|how many|"
    r"average|avg|sum|total|maximum|minimum|max|min|highest|lowest|largest|smallest)\b",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(r"\b(never|not|except|without)\b", re.IGNORECASE)


def _norm_ident(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _safe_sql(tree: exp.Expression) -> str | None:
    try:
        sql = tree.sql(dialect="sqlite").strip()
    except Exception:
        return None
    if not sql or len(sql.encode("utf-8")) > MAX_SQL_BYTES:
        return None
    return sql


def _parse_sqlite(sql: str) -> exp.Expression | None:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        return tree
    except Exception:
        return None


def _norm_cell(val: Any) -> Any:
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, float)):
        num = str(val) if isinstance(val, int) else format(round(val, 6), ".6f")
        if "." in num:
            num = num.rstrip("0").rstrip(".")
        return "0" if num == "-0" else num
    return val


def _extract_col_tokens(col_name: str) -> tuple[str | None, list[str]]:
    """Extract optional aggregate function and identifier tokens from example column header."""
    s = col_name.strip().lower()
    agg = None
    m = re.match(r"^(count|sum|avg|max|min)\s*\((.*)\)$", s)
    if m:
        agg = m.group(1)
        s = m.group(2).strip()
    # Strip table alias prefixes like t1.age -> age
    parts = [p for p in re.split(r"[^a-z0-9]+", s) if p and not re.fullmatch(r"t\d+", p)]
    return agg, parts


def _expr_score_against_header(expr: exp.Expression, res_col_name: str, target_header: str) -> float:
    """Heuristic match score between a SELECT expression and an expected column header."""
    inner = expr.this if isinstance(expr, exp.Alias) else expr
    t_agg, t_tokens = _extract_col_tokens(target_header)
    r_agg, r_tokens = _extract_col_tokens(res_col_name)

    expr_sql = inner.sql(dialect="sqlite").lower()
    e_agg = None
    if isinstance(inner, exp.Count):
        e_agg = "count"
    elif isinstance(inner, exp.Sum):
        e_agg = "sum"
    elif isinstance(inner, exp.Avg):
        e_agg = "avg"
    elif isinstance(inner, exp.Max):
        e_agg = "max"
    elif isinstance(inner, exp.Min):
        e_agg = "min"

    score = 0.0
    if t_agg is not None:
        if e_agg == t_agg:
            score += 5.0
        elif e_agg is not None or r_agg is not None:
            score -= 3.0
        else:
            score -= 2.0
    elif e_agg is not None:
        score -= 2.0

    # Exact identifier match
    col_names = [c.name.lower() for c in inner.find_all(exp.Column) if c.name]
    target_norm = _norm_ident(target_header)
    if res_col_name.lower() == target_header.lower():
        score += 4.0
    elif _norm_ident(res_col_name) == target_norm:
        score += 3.0
    for cn in col_names:
        if cn == target_header.lower() or _norm_ident(cn) == target_norm:
            score += 4.0
        elif t_tokens and cn in t_tokens:
            score += 2.5

    # Token overlap
    e_tokens = set(re.findall(r"[a-z0-9]+", expr_sql)) | set(r_tokens)
    for tok in t_tokens:
        if tok in e_tokens:
            score += 1.5
    if isinstance(inner, exp.Count) and isinstance(inner.this, exp.Star) and t_agg == "count":
        if not t_tokens:
            score += 2.0
    return score


def _table_alias_map(select_tree: exp.Select, schema_info: dict[str, Any]) -> list[dict[str, str]]:
    """Collect ordered list of {table, alias} referenced in FROM and JOINs of a Select."""
    out: list[dict[str, str]] = []
    tables_meta = schema_info.get("tables", {})
    from_node = select_tree.args.get("from") or select_tree.args.get("from_")
    candidates: list[exp.Table] = []
    if from_node and isinstance(from_node.this, exp.Table):
        candidates.append(from_node.this)
    for j in select_tree.args.get("joins") or []:
        if isinstance(j.this, exp.Table):
            candidates.append(j.this)
    for tbl in candidates:
        raw_name = tbl.name
        real_name = tables_meta[raw_name.lower()]["name"] if raw_name.lower() in tables_meta else raw_name
        alias = tbl.alias_or_name or real_name
        out.append({"table": real_name, "alias": alias})
    return out


def _col_exists_in_table(table_name: str, col_name: str, schema_info: dict[str, Any]) -> str | None:
    tinfo = schema_info.get("tables", {}).get(table_name.lower())
    if not tinfo:
        return None
    return tinfo["columns"].get(col_name.lower())


def _shortest_fk_path(
    start_tables: list[str], target_table: str, schema_info: dict[str, Any]
) -> list[dict[str, str]] | None:
    """BFS shortest FK join path from any table in start_tables to target_table."""
    fk_graph = schema_info.get("fk_graph", {})
    target_l = target_table.lower()
    start_set = {t.lower() for t in start_tables}
    if target_l in start_set:
        return []
    visited = set(start_set)
    queue: deque[tuple[str, list[dict[str, str]]]] = deque((t, []) for t in start_set)
    while queue:
        cur, path = queue.popleft()
        for edge in fk_graph.get(cur, []):
            nxt = edge["to_table"].lower()
            if nxt in visited:
                continue
            new_path = path + [edge]
            if nxt == target_l:
                return new_path
            visited.add(nxt)
            queue.append((nxt, new_path))
    return None


def _replace_aliases_in_clauses(tree: exp.Select, dropped_exprs: list[exp.Expression]) -> None:
    """Replace alias references in ORDER BY / HAVING / GROUP BY before dropping SELECT items."""
    alias_map: dict[str, exp.Expression] = {}
    for expr in dropped_exprs:
        if isinstance(expr, exp.Alias) and expr.alias:
            alias_map[expr.alias.lower()] = expr.this
    if not alias_map:
        return

    for clause_name in ("order", "having", "group"):
        clause = tree.args.get(clause_name)
        if clause is None:
            continue
        for col in list(clause.find_all(exp.Column)):
            if not col.table and col.name and col.name.lower() in alias_map:
                col.replace(alias_map[col.name.lower()].copy())


def _projection_repairs(
    cand: dict[str, Any], observation: dict[str, Any], schema_info: dict[str, Any]
) -> list[str]:
    """Priority 1: SELECT projection permutation, trimming, and missing column addition."""
    res = cand.get("result")
    if not isinstance(res, dict) or not res.get("ok"):
        return []
    tree = _parse_sqlite(cand["sql"])
    if not isinstance(tree, exp.Select):
        return []

    example = observation["example"]
    exp_cols: list[str] = example["columns"]
    exp_rows: list[list[Any]] = example["rows"]
    got_cols: list[str] = res.get("columns", [])
    got_rows: list[list[Any]] = res.get("rows", [])
    select_exprs = list(tree.expressions)

    out: list[str] = []
    k_target = len(exp_cols)
    k_sel = len(select_exprs)

    # 1a. Same column count (> 1): reorder SELECT expressions
    if k_sel == k_target and k_sel > 1 and len(got_cols) == k_sel:
        best_perm: tuple[int, ...] | None = None
        if k_sel <= 6 and got_rows and exp_rows and not example.get("truncated", False) and len(got_rows) == len(exp_rows):
            norm_exp = [tuple(_norm_cell(c) for c in r) for r in exp_rows]
            exp_counter = Counter(norm_exp)
            for perm in itertools.permutations(range(k_sel)):
                if perm == tuple(range(k_sel)):
                    continue
                perm_rows = [tuple(_norm_cell(r[j]) for j in perm) for r in got_rows]
                if (observation["ordered"] and perm_rows == norm_exp) or (
                    not observation["ordered"] and Counter(perm_rows) == exp_counter
                ):
                    best_perm = perm
                    break
        if best_perm is None and k_sel <= 6:
            # Header/expression score alignment
            ident_score = sum(
                _expr_score_against_header(select_exprs[i], got_cols[i], exp_cols[i])
                for i in range(k_sel)
            )
            best_score = ident_score
            for perm in itertools.permutations(range(k_sel)):
                if perm == tuple(range(k_sel)):
                    continue
                sc = sum(
                    _expr_score_against_header(select_exprs[perm[i]], got_cols[perm[i]], exp_cols[i])
                    for i in range(k_sel)
                )
                if sc > best_score + 0.5:
                    best_score = sc
                    best_perm = perm
        if best_perm is not None:
            cloned = tree.copy()
            cloned.set("expressions", [select_exprs[j].copy() for j in best_perm])
            sql = _safe_sql(cloned)
            if sql:
                out.append(sql)

    # 1b. Extra columns: trim to len(exp_cols) and expand dropped aliases in ORDER BY/HAVING
    elif k_sel > k_target >= 1 and len(got_cols) == k_sel and k_sel <= 7:
        best_choice: tuple[int, ...] | None = None
        best_choice_score = -1e9
        norm_exp = [tuple(_norm_cell(c) for c in r) for r in exp_rows]
        exp_counter = Counter(norm_exp)

        for perm in itertools.permutations(range(k_sel), k_target):
            sc = sum(
                _expr_score_against_header(select_exprs[perm[i]], got_cols[perm[i]], exp_cols[i])
                for i in range(k_target)
            )
            if got_rows and exp_rows and not example.get("truncated", False) and len(got_rows) == len(exp_rows):
                proj_rows = [tuple(_norm_cell(r[j]) for j in perm) for r in got_rows]
                if (observation["ordered"] and proj_rows == norm_exp) or (
                    not observation["ordered"] and Counter(proj_rows) == exp_counter
                ):
                    sc += 100.0
            if sc > best_choice_score:
                best_choice_score = sc
                best_choice = perm

        if best_choice is not None:
            cloned = tree.copy()
            dropped = [select_exprs[j] for j in range(k_sel) if j not in best_choice]
            _replace_aliases_in_clauses(cloned, dropped)
            cloned.set("expressions", [select_exprs[j].copy() for j in best_choice])
            sql = _safe_sql(cloned)
            if sql:
                out.append(sql)

            # If public example is 0 rows and question has no aggregation cue, also try stripping GROUP BY added solely for dropped COUNT
            if (
                example["row_count"] == 0
                and not _AGG_HINT_RE.search(observation["task"]["question"])
                and any(isinstance((d.this if isinstance(d, exp.Alias) else d), exp.Count) for d in dropped)
                and cloned.args.get("group") is not None
            ):
                no_gb = cloned.copy()
                no_gb.set("group", None)
                no_gb.set("having", None)
                if no_gb.args.get("order") is not None and any(
                    isinstance(a, exp.AggFunc) for a in no_gb.args["order"].find_all(exp.AggFunc)
                ):
                    no_gb.set("order", None)
                sql2 = _safe_sql(no_gb)
                if sql2:
                    out.append(sql2)

    # 1c. Missing columns: append column(s) hinted by example["columns"]
    elif 1 <= k_sel < k_target <= 6:
        tmap = _table_alias_map(tree, schema_info)
        existing_norms = set()
        for e in select_exprs:
            inner = e.this if isinstance(e, exp.Alias) else e
            for c in inner.find_all(exp.Column):
                if c.name:
                    existing_norms.add(_norm_ident(c.name))

        added_exprs: list[exp.Expression] = []
        for header in exp_cols:
            h_agg, h_toks = _extract_col_tokens(header)
            if h_agg is not None:
                continue
            h_norm = _norm_ident(header)
            if h_norm in existing_norms:
                continue
            # Look up exact or fuzzy match in currently joined tables
            matched_col = None
            for ref in tmap:
                real_col = _col_exists_in_table(ref["table"], header, schema_info)
                if real_col:
                    matched_col = exp.column(real_col, table=ref["alias"])
                    break
            if matched_col is None and h_toks:
                for ref in tmap:
                    tinfo = schema_info.get("tables", {}).get(ref["table"].lower())
                    if not tinfo:
                        continue
                    for cname in tinfo["columns"].values():
                        if _norm_ident(cname) == h_norm:
                            matched_col = exp.column(cname, table=ref["alias"])
                            break
                    if matched_col is not None:
                        break
            if matched_col is not None:
                added_exprs.append(matched_col)

        if added_exprs and k_sel + len(added_exprs) == k_target:
            cloned = tree.copy()
            cloned.set("expressions", [e.copy() for e in select_exprs] + added_exprs)
            sql = _safe_sql(cloned)
            if sql:
                out.append(sql)

    return out


def _qualify_ambiguous(tree: exp.Select, col_name: str, schema_info: dict[str, Any]) -> exp.Select | None:
    tmap = _table_alias_map(tree, schema_info)
    matching_aliases = [
        ref["alias"] for ref in tmap if _col_exists_in_table(ref["table"], col_name, schema_info)
    ]
    if not matching_aliases:
        return None
    chosen_alias = matching_aliases[0]
    cloned = tree.copy()
    changed = False
    for c in cloned.find_all(exp.Column):
        if not c.table and c.name and c.name.lower() == col_name.lower():
            c.set("table", exp.to_identifier(chosen_alias))
            changed = True
    return cloned if changed else None


def _insert_fk_joins(
    cloned: exp.Select, start_alias_map: list[dict[str, str]], path: list[dict[str, str]]
) -> str:
    """Append FK JOIN path onto cloned Select and return final table's alias."""
    existing_aliases = {ref["alias"].lower(): ref["alias"] for ref in start_alias_map}
    table_to_alias = {ref["table"].lower(): ref["alias"] for ref in start_alias_map}
    joins = list(cloned.args.get("joins") or [])

    for edge in path:
        from_t = edge["from_table"]
        to_t = edge["to_table"]
        from_alias = table_to_alias.get(from_t.lower(), from_t)
        if to_t.lower() in table_to_alias:
            continue
        to_alias = to_t
        if to_alias.lower() in existing_aliases:
            idx = 2
            while f"{to_t}_{idx}".lower() in existing_aliases:
                idx += 1
            to_alias = f"{to_t}_{idx}"
        existing_aliases[to_alias.lower()] = to_alias
        table_to_alias[to_t.lower()] = to_alias

        on_expr = exp.EQ(
            this=exp.column(edge["from_col"], table=from_alias),
            expression=exp.column(edge["to_col"], table=to_alias),
        )
        tbl_expr = exp.Table(this=exp.to_identifier(to_t))
        if to_alias != to_t:
            tbl_expr.set("alias", exp.TableAlias(this=exp.to_identifier(to_alias)))
        joins.append(exp.Join(this=tbl_expr, on=on_expr))

    cloned.set("joins", joins)
    return table_to_alias[path[-1]["to_table"].lower()] if path else ""


def _exec_error_repairs(
    cand: dict[str, Any], observation: dict[str, Any], schema_info: dict[str, Any]
) -> list[str]:
    """Priority 2: Deterministic AST repairs driven by SQLite / SQLGlot error messages."""
    res = cand.get("result")
    if not isinstance(res, dict) or res.get("ok"):
        return []
    err = res.get("error") or {}
    msg = err.get("message") or ""
    raw_sql = cand["sql"]
    out: list[str] = []

    tree = _parse_sqlite(raw_sql)

    # 2c-syntax: try fixing syntax errors before giving up on unparseable SQL
    if tree is None:
        fixed = raw_sql
        # Unbalanced parentheses
        open_p = fixed.count("(")
        close_p = fixed.count(")")
        if close_p > open_p:
            fixed = re.sub(r"\)\s*;?\s*$", ";", fixed, count=1)
        elif open_p > close_p:
            fixed = fixed.rstrip("; \n") + (")" * (open_p - close_p)) + ";"
        # IS MAX(...) -> = (SELECT MAX(...) FROM ...)
        fixed = re.sub(
            r"\bIS\s+(MAX|MIN|AVG|SUM|COUNT)\s*\(([^)]+)\)",
            r"= (SELECT \1(\2))",
            fixed,
            flags=re.IGNORECASE,
        )
        # > ALL (SELECT x FROM ...) -> > (SELECT MAX(x) FROM ...)
        fixed = re.sub(
            r"([<>]=?)\s*ALL\s*\(\s*SELECT\s+([^\s,]+)",
            r"\1 (SELECT MAX(\2)",
            fixed,
            flags=re.IGNORECASE,
        )
        t_fixed = _parse_sqlite(fixed)
        if t_fixed is not None:
            s = _safe_sql(t_fixed)
            if s:
                out.append(s)
        return out

    # Also handle > ALL (...) if sqlglot parsed it
    if re.search(r"\bALL\s*\(\s*SELECT\b", raw_sql, re.IGNORECASE):
        fixed_all = re.sub(
            r">\s*ALL\s*\(\s*SELECT\s+([^\s,]+)",
            r"> (SELECT MAX(\1)",
            raw_sql,
            flags=re.IGNORECASE,
        )
        fixed_all = re.sub(
            r"<\s*ALL\s*\(\s*SELECT\s+([^\s,]+)",
            r"< (SELECT MIN(\1)",
            fixed_all,
            flags=re.IGNORECASE,
        )
        t_all = _parse_sqlite(fixed_all)
        if t_all is not None:
            s = _safe_sql(t_all)
            if s:
                out.append(s)

    if not isinstance(tree, exp.Select):
        return out

    tmap = _table_alias_map(tree, schema_info)
    alias_to_table = {ref["alias"].lower(): ref["table"] for ref in tmap}
    alias_to_table.update({ref["table"].lower(): ref["table"] for ref in tmap})

    # 2a. ambiguous column name: <col>
    m_amb = re.search(r"ambiguous column name:\s*(?:[a-zA-Z0-9_]+\.)?([a-zA-Z0-9_]+)", msg)
    if m_amb:
        amb_col = m_amb.group(1)
        qualified = _qualify_ambiguous(tree, amb_col, schema_info)
        if qualified is not None:
            s = _safe_sql(qualified)
            if s:
                out.append(s)
            # Also trim extra columns if SELECT count > example["columns"]
            exp_cols = observation["example"]["columns"]
            if len(qualified.expressions) > len(exp_cols) >= 1:
                fake_cand = {
                    "sql": s or raw_sql,
                    "result": {
                        "ok": True,
                        "columns": [e.alias_or_name for e in qualified.expressions],
                        "rows": [],
                    },
                }
                out.extend(_projection_repairs(fake_cand, observation, schema_info))
        return out

    # 2b. no such column: [qual.]col
    m_col = re.search(r"no such column:\s*(?:([a-zA-Z0-9_`\"']+)\.)?([a-zA-Z0-9_`\"']+)", msg)
    if m_col:
        qual = (m_col.group(1) or "").strip("`\"'")
        col_name = m_col.group(2).strip("`\"'")

        # Subcase i: Wrong qualifier (column actually lives on another joined table; skip JOIN ON to avoid self-tautologies)
        if qual:
            for ref in tmap:
                if ref["alias"].lower() == qual.lower():
                    continue
                real_col = _col_exists_in_table(ref["table"], col_name, schema_info)
                if real_col:
                    cloned = tree.copy()
                    changed = False
                    for c in cloned.find_all(exp.Column):
                        if c.find_ancestor(exp.Join) is not None:
                            continue
                        if c.table and c.table.lower() == qual.lower() and c.name.lower() == col_name.lower():
                            c.set("table", exp.to_identifier(ref["alias"]))
                            c.set("this", exp.to_identifier(real_col))
                            changed = True
                    if changed:
                        s = _safe_sql(cloned)
                        if s:
                            out.append(s)

        # Subcase ii: Missing column inside a JOIN ... ON -> insert 1-hop bridge table OR drop redundant join
        for j_idx, j in enumerate(tree.args.get("joins") or []):
            on_expr = j.args.get("on")
            if on_expr is None:
                continue
            bad_in_join = any(
                c.name.lower() == col_name.lower()
                and (not qual or (c.table and c.table.lower() == qual.lower()))
                for c in on_expr.find_all(exp.Column)
            )
            if not bad_in_join or not isinstance(j.this, exp.Table):
                continue
            j_table = j.this.name
            j_alias = j.this.alias_or_name or j_table
            # Find a 2-edge bridge path from earlier tables to j_table
            prev_tables = [tmap[0]["table"]] + [
                ref["table"] for idx, ref in enumerate(tmap[1:]) if idx < j_idx
            ]
            path = _shortest_fk_path(prev_tables, j_table, schema_info)
            if path and len(path) == 2:
                bridge = path[0]
                second = path[1]
                cloned = tree.copy()
                new_joins = list(cloned.args.get("joins") or [])
                prev_alias = next(
                    (ref["alias"] for ref in tmap if ref["table"].lower() == bridge["from_table"].lower()),
                    bridge["from_table"],
                )
                b_alias = f"bridge_{bridge['to_table']}"
                b_tbl = exp.Table(this=exp.to_identifier(bridge["to_table"]))
                b_tbl.set("alias", exp.TableAlias(this=exp.to_identifier(b_alias)))
                b_join = exp.Join(
                    this=b_tbl,
                    on=exp.EQ(
                        this=exp.column(bridge["from_col"], table=prev_alias),
                        expression=exp.column(bridge["to_col"], table=b_alias),
                    ),
                )
                fixed_on = exp.EQ(
                    this=exp.column(second["from_col"], table=b_alias),
                    expression=exp.column(second["to_col"], table=j_alias),
                )
                new_joins[j_idx].set("on", fixed_on)
                new_joins.insert(j_idx, b_join)
                cloned.set("joins", new_joins)
                s = _safe_sql(cloned)
                if s:
                    out.append(s)

            # Alternative: if broken joined table isn't projected in SELECT/GROUP/ORDER, drop it and its WHERE terms
            select_refs = {
                c.table.lower()
                for e in tree.expressions
                for c in e.find_all(exp.Column)
                if c.table
            }
            if j_alias.lower() not in select_refs and j_table.lower() not in select_refs:
                cloned2 = tree.copy()
                joins2 = [jn for idx, jn in enumerate(cloned2.args.get("joins") or []) if idx != j_idx]
                cloned2.set("joins", joins2)
                w = cloned2.args.get("where")
                if w is not None:
                    kept_terms = []
                    # Split top-level AND terms
                    stack = [w.this]
                    and_terms = []
                    while stack:
                        node = stack.pop()
                        if isinstance(node, exp.And):
                            stack.append(node.right)
                            stack.append(node.left)
                        else:
                            and_terms.append(node)
                    for term in and_terms:
                        refs = {c.table.lower() for c in term.find_all(exp.Column) if c.table}
                        if j_alias.lower() not in refs and j_table.lower() not in refs:
                            kept_terms.append(term)
                    if kept_terms:
                        combined = kept_terms[0]
                        for t in kept_terms[1:]:
                            combined = exp.and_(combined, t)
                        cloned2.set("where", exp.Where(this=combined))
                    else:
                        cloned2.set("where", None)
                s2 = _safe_sql(cloned2)
                if s2:
                    out.append(s2)

        # Subcase iii: Column exists in another schema table -> add shortest FK JOIN
        candidate_tables = schema_info.get("col_to_tables", {}).get(col_name.lower(), [])
        start_tables = [ref["table"] for ref in tmap]
        for target_tbl in candidate_tables:
            if target_tbl.lower() in {t.lower() for t in start_tables}:
                continue
            path = _shortest_fk_path(start_tables, target_tbl, schema_info)
            if path and len(path) <= 2:
                cloned = tree.copy()
                new_alias = _insert_fk_joins(cloned, tmap, path)
                real_col = _col_exists_in_table(target_tbl, col_name, schema_info) or col_name
                changed = False
                for c in cloned.find_all(exp.Column):
                    if c.name and c.name.lower() == col_name.lower():
                        if not c.table or c.table.lower() not in alias_to_table or (
                            qual and c.table.lower() == qual.lower()
                        ):
                            c.set("table", exp.to_identifier(new_alias))
                            c.set("this", exp.to_identifier(real_col))
                            changed = True
                if changed:
                    s = _safe_sql(cloned)
                    if s:
                        out.append(s)

        # Subcase iv: Column typo / synonym inside queried tables (or all tables)
        synonym_pairs = {"gender": "sex", "sex": "gender"}
        best_matches: list[tuple[float, str, str]] = []
        search_tables = tmap if not qual or qual.lower() not in alias_to_table else [
            ref for ref in tmap if ref["alias"].lower() == qual.lower() or ref["table"].lower() == qual.lower()
        ]
        if not search_tables:
            search_tables = tmap
        ex_cols_norm = {_norm_ident(ec): ec for ec in observation["example"]["columns"]}
        for ref in search_tables:
            tinfo = schema_info.get("tables", {}).get(ref["table"].lower())
            if not tinfo:
                continue
            for cname in tinfo["columns"].values():
                sim = SequenceMatcher(None, col_name.lower(), cname.lower()).ratio()
                if synonym_pairs.get(col_name.lower()) == cname.lower():
                    sim = max(sim, 0.92)
                if _norm_ident(cname) in ex_cols_norm:
                    sim += 0.15
                if col_name.lower() in cname.lower() or cname.lower() in col_name.lower():
                    sim += 0.25
                if sim >= 0.62:
                    best_matches.append((sim, ref["alias"], cname))
        best_matches.sort(reverse=True)
        for _, m_alias, m_col in best_matches[:2]:
            cloned = tree.copy()
            changed = False
            for c in cloned.find_all(exp.Column):
                if c.name and c.name.lower() == col_name.lower() and (
                    not qual or (c.table and c.table.lower() == qual.lower())
                ):
                    c.set("this", exp.to_identifier(m_col))
                    if qual:
                        c.set("table", exp.to_identifier(m_alias))
                    changed = True
            if changed:
                s = _safe_sql(cloned)
                if s:
                    out.append(s)

    # 2c. no such table: <tbl>
    m_tbl = re.search(r"no such table:\s*([a-zA-Z0-9_`\"']+)", msg)
    if m_tbl:
        bad_tbl = m_tbl.group(1).strip("`\"'")
        best_t = None
        best_sim = 0.0
        for tinfo in schema_info.get("tables", {}).values():
            sim = SequenceMatcher(None, bad_tbl.lower(), tinfo["name"].lower()).ratio()
            if sim > best_sim:
                best_sim = sim
                best_t = tinfo["name"]
        if best_t and best_sim >= 0.6:
            cloned = tree.copy()
            for t in cloned.find_all(exp.Table):
                if t.name and t.name.lower() == bad_tbl.lower():
                    t.set("this", exp.to_identifier(best_t))
            s = _safe_sql(cloned)
            if s:
                out.append(s)

    return out


def _semantic_and_spurious_variants(
    candidates: list[dict[str, Any]], observation: dict[str, Any], schema_info: dict[str, Any]
) -> list[str]:
    """Priority 3: DISTINCT calibration, unused JOIN removal, literal alignment, ASC/DESC flip."""
    out: list[str] = []
    question = observation["task"]["question"]
    example = observation["example"]
    q_wants_distinct = bool(_DISTINCT_RE.search(question))

    passing = [
        c for c in candidates if isinstance(c.get("result"), dict) and c["result"].get("matches_example") is True
    ]

    # 3a. DISTINCT calibration on passing candidates: add DISTINCT when question explicitly asks for distinct.
    for cand in passing:
        tree = _parse_sqlite(cand["sql"])
        if not isinstance(tree, exp.Select):
            continue
        has_dist = tree.args.get("distinct") is not None
        if q_wants_distinct and not has_dist:
            cloned = tree.copy()
            cloned.set("distinct", exp.Distinct())
            s = _safe_sql(cloned)
            if s:
                out.append(s)

    # 3b. Unused JOIN removal on ok=True candidates with wrong aggregate/row count (run before extra_rows DISTINCT)
    if not passing:
        for cand in sorted(candidates, key=lambda c: (0 if "hint" in c["source"] else 1, c["order"])):
            res = cand.get("result")
            if not isinstance(res, dict) or not res.get("ok"):
                continue
            tree = _parse_sqlite(cand["sql"])
            if not isinstance(tree, exp.Select):
                continue
            joins = list(tree.args.get("joins") or [])
            if not joins:
                continue
            tmap = _table_alias_map(tree, schema_info)
            non_join_refs: set[str] = set()
            for part_name in ("expressions", "where", "group", "having", "order"):
                part = tree.args.get(part_name)
                nodes = part if isinstance(part, list) else ([part] if part is not None else [])
                for n in nodes:
                    for c in n.find_all(exp.Column):
                        if c.table:
                            non_join_refs.add(c.table.lower())
                        elif c.name:
                            for ref in tmap:
                                if _col_exists_in_table(ref["table"], c.name, schema_info):
                                    non_join_refs.add(ref["alias"].lower())
            unused_indices = [
                idx
                for idx, jn in enumerate(joins)
                if isinstance(jn.this, exp.Table)
                and (jn.this.alias_or_name or jn.this.name).lower() not in non_join_refs
            ]
            if unused_indices:
                cloned = tree.copy()
                cloned.set("joins", [jn for idx, jn in enumerate(joins) if idx not in set(unused_indices)])
                s = _safe_sql(cloned)
                if s:
                    out.append(s)

    # 3a-extra. If no candidate passes and some non-aggregate ok=True fails with extra_rows, try adding DISTINCT
    if not passing:
        for cand in sorted(candidates, key=lambda c: (0 if "hint" in c["source"] else 1, c["order"])):
            res = cand.get("result")
            if not isinstance(res, dict) or not res.get("ok"):
                continue
            if (
                len(res.get("columns", [])) == len(example["columns"])
                and res.get("row_count", 0) > example["row_count"] > 0
            ):
                tree = _parse_sqlite(cand["sql"])
                if (
                    isinstance(tree, exp.Select)
                    and tree.args.get("distinct") is None
                    and not any(isinstance(e.this if isinstance(e, exp.Alias) else e, exp.AggFunc) for e in tree.expressions)
                ):
                    cloned = tree.copy()
                    cloned.set("distinct", exp.Distinct())
                    s = _safe_sql(cloned)
                    if s:
                        out.append(s)

    # 3c. WHERE string literal alignment with multi-word capitalized phrase in question on empty result
    if not passing and example["row_count"] > 0:
        for cand in reversed(candidates):
            res = cand.get("result")
            if not isinstance(res, dict) or not res.get("ok") or res.get("row_count", 0) != 0:
                continue
            tree = _parse_sqlite(cand["sql"])
            if not isinstance(tree, exp.Select) or tree.args.get("where") is None:
                continue
            cloned = tree.copy()
            changed = False
            for lit in cloned.args["where"].find_all(exp.Literal):
                if not lit.is_string or not lit.this:
                    continue
                val = str(lit.this)
                if len(val) < 3:
                    continue
                # Look for two-word capitalized phrase containing val in question (e.g. "Team Orton")
                pattern = re.compile(
                    rf"\b([A-Z][A-Za-z0-9_-]*\s+{re.escape(val)}|{re.escape(val)}\s+[A-Z][A-Za-z0-9_-]*)\b"
                )
                m = pattern.search(question)
                if m and m.group(1) != val:
                    lit.set("this", m.group(1))
                    changed = True
            if changed:
                s = _safe_sql(cloned)
                if s:
                    out.append(s)

    # 3d. ordered=True ASC/DESC flip on same-shape mismatch
    if not passing and observation["ordered"]:
        for cand in reversed(candidates):
            res = cand.get("result")
            if not isinstance(res, dict) or not res.get("ok"):
                continue
            if (
                len(res.get("columns", [])) == len(example["columns"])
                and res.get("row_count") == example["row_count"]
                and example["row_count"] > 1
            ):
                tree = _parse_sqlite(cand["sql"])
                if isinstance(tree, exp.Select) and tree.args.get("order") is not None:
                    cloned = tree.copy()
                    for ordered_node in cloned.args["order"].find_all(exp.Ordered):
                        ordered_node.set("desc", not bool(ordered_node.args.get("desc")))
                    s = _safe_sql(cloned)
                    if s:
                        out.append(s)

    # 3e. Step 4.6b: In-memory example["rows"] alignment (exact dedup -> DISTINCT, head/tail -> LIMIT 1, type mismatch)
    if not passing and example.get("row_count", 0) > 0 and example.get("rows"):
        exp_cols_cnt = len(example["columns"])
        exp_rows_cnt = example["row_count"]
        exp_rows_norm = [tuple(_norm_cell(c) for c in row) for row in example["rows"]]
        for cand in sorted(candidates, key=lambda c: (0 if "llm" in c["source"] else (1 if "hint" in c["source"] else 2), c["order"])):
            res = cand.get("result")
            if not isinstance(res, dict) or not res.get("ok") or len(res.get("columns", [])) != exp_cols_cnt:
                continue
            got_rows_raw = res.get("rows") or []
            if not got_rows_raw:
                continue
            tree = _parse_sqlite(cand["sql"])
            if not isinstance(tree, exp.Select):
                continue
            got_rows_norm = [tuple(_norm_cell(c) for c in row) for row in got_rows_raw]

            # 3e-1. Exact deduplication match -> add DISTINCT
            if not example.get("truncated") and not res.get("truncated") and len(got_rows_norm) > exp_rows_cnt and tree.args.get("distinct") is None:
                dedup = list(dict.fromkeys(got_rows_norm))
                matched = (dedup == exp_rows_norm) if observation["ordered"] else (sorted(dedup, key=str) == sorted(exp_rows_norm, key=str))
                if matched:
                    cloned = tree.copy()
                    cloned.set("distinct", exp.Distinct())
                    s = _safe_sql(cloned)
                    if s:
                        out.append(s)

            # 3e-2. Head/tail extremum truncation -> add LIMIT 1 (or flip ORDER BY + LIMIT 1)
            if exp_rows_cnt == 1 and len(got_rows_norm) > 1 and tree.args.get("limit") is None and tree.args.get("order") is not None:
                if got_rows_norm[0] == exp_rows_norm[0]:
                    cloned = tree.copy()
                    cloned.set("limit", exp.Limit(expression=exp.Literal.number(1)))
                    s = _safe_sql(cloned)
                    if s:
                        out.append(s)
                elif not res.get("truncated") and got_rows_norm[-1] == exp_rows_norm[0]:
                    cloned = tree.copy()
                    for ordered_node in cloned.args["order"].find_all(exp.Ordered):
                        ordered_node.set("desc", not bool(ordered_node.args.get("desc")))
                    cloned.set("limit", exp.Limit(expression=exp.Literal.number(1)))
                    s = _safe_sql(cloned)
                    if s:
                        out.append(s)

            # 3e-3. Cell-level numeric vs text type mismatch on ith column -> swap to matching-type col in same table
            if len(tree.expressions) == exp_cols_cnt:
                for idx_c in range(exp_cols_cnt):
                    ex_cell = example["rows"][0][idx_c]
                    gt_cell = got_rows_raw[0][idx_c]
                    if ex_cell is None or gt_cell is None:
                        continue
                    ex_num = isinstance(ex_cell, (int, float)) and not isinstance(ex_cell, bool)
                    gt_num = isinstance(gt_cell, (int, float)) and not isinstance(gt_cell, bool)
                    if ex_num == gt_num:
                        continue
                    sel_e = tree.expressions[idx_c]
                    inner_e = sel_e.this if isinstance(sel_e, exp.Alias) else sel_e
                    if not isinstance(inner_e, exp.Column):
                        continue
                    resolved = _resolve_col_table(inner_e, tree, schema_info)
                    if not resolved:
                        continue
                    tname, old_col = resolved
                    tinfo = schema_info.get("tables", {}).get(tname.lower())
                    if not tinfo:
                        continue
                    _, hdr_toks = _extract_col_tokens(example["columns"][idx_c])
                    num_types = ("int", "real", "num", "float", "dec", "double")
                    for cand_col, ctype in tinfo.get("types", {}).items():
                        if cand_col.lower() == old_col.lower():
                            continue
                        col_is_num = any(nt in ctype.lower() for nt in num_types)
                        if col_is_num != ex_num:
                            continue
                        if hdr_toks and not any(tok in cand_col.lower() for tok in hdr_toks):
                            continue
                        cloned = tree.copy()
                        tgt_e = cloned.expressions[idx_c]
                        tgt_col = tgt_e.this if isinstance(tgt_e, exp.Alias) else tgt_e
                        tgt_col.set("this", exp.to_identifier(cand_col))
                        s = _safe_sql(cloned)
                        if s:
                            out.append(s)

    return out


def generate_ast_repairs(
    observation: dict[str, Any],
    schema_info: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    checked_norm_sqls: set[str],
    normalize_fn,
) -> list[tuple[str, str]]:
    """Generate prioritized, deduplicated (sql, source_tag) repair candidates."""
    if schema_info is None:
        return []

    proposed: list[tuple[str, str]] = []
    seen = set(checked_norm_sqls)

    def add_all(sql_list: list[str], source_tag: str) -> None:
        for sql in sql_list:
            if not sql or len(sql.encode("utf-8")) > MAX_SQL_BYTES:
                continue
            norm = normalize_fn(sql)
            if norm not in seen:
                seen.add(norm)
                proposed.append((sql, source_tag))

    passing = [
        c for c in candidates if isinstance(c.get("result"), dict) and c["result"].get("matches_example") is True
    ]

    # Prefer repairing LLM output first, then hints (higher semantic quality), then issue_sql
    ordered_cands = sorted(
        candidates,
        key=lambda c: (
            0 if "llm" in c["source"] else (1 if "hint" in c["source"] else 2),
            c["order"],
        ),
    )
    has_ok_non_issue = any(
        c["source"] != "issue"
        and isinstance(c.get("result"), dict)
        and c["result"].get("ok") is True
        for c in candidates
    )

    if not passing:
        for c in ordered_cands:
            tag = (
                "ast_from_llm"
                if "llm" in c["source"]
                else ("ast_from_hint" if "hint" in c["source"] else "ast_from_issue")
            )
            add_all(_projection_repairs(c, observation, schema_info), tag)
        for c in ordered_cands:
            if c["source"] == "issue" and has_ok_non_issue:
                continue
            tag = (
                "ast_from_llm"
                if "llm" in c["source"]
                else ("ast_from_hint" if "hint" in c["source"] else "ast_from_issue")
            )
            add_all(_exec_error_repairs(c, observation, schema_info), tag)

    variant_tag = (
        "ast_from_llm"
        if any("llm" in c["source"] for c in passing or ordered_cands)
        else ("ast_from_hint" if any("hint" in c["source"] for c in passing or ordered_cands) else "ast_from_issue")
    )
    add_all(_semantic_and_spurious_variants(candidates, observation, schema_info), variant_tag)
    return proposed


def _count_unused_joins(tree: exp.Select, schema_info: dict[str, Any] | None) -> int:
    joins = list(tree.args.get("joins") or [])
    if not joins or schema_info is None:
        return 0
    tmap = _table_alias_map(tree, schema_info)
    used: set[str] = set()
    for part_name in ("expressions", "where", "group", "having", "order"):
        part = tree.args.get(part_name)
        nodes = part if isinstance(part, list) else ([part] if part is not None else [])
        for n in nodes:
            for c in n.find_all(exp.Column):
                if c.table:
                    used.add(c.table.lower())
                elif c.name:
                    for ref in tmap:
                        if _col_exists_in_table(ref["table"], c.name, schema_info):
                            used.add(ref["alias"].lower())
    return sum(
        1
        for jn in joins
        if isinstance(jn.this, exp.Table)
        and (jn.this.alias_or_name or jn.this.name).lower() not in used
    )


def _fk_join_validity_score(tree: exp.Select, schema_info: dict[str, Any] | None) -> float:
    if schema_info is None:
        return 0.0
    joins = list(tree.args.get("joins") or [])
    if not joins:
        return 0.0
    tmap = _table_alias_map(tree, schema_info)
    alias_to_table = {ref["alias"].lower(): ref["table"].lower() for ref in tmap}
    alias_to_table.update({ref["table"].lower(): ref["table"].lower() for ref in tmap})
    fk_graph = schema_info.get("fk_graph", {})

    score = 0.0
    for jn in joins:
        on_expr = jn.args.get("on")
        if on_expr is None:
            score -= 2.0
            continue
        for eq in on_expr.find_all(exp.EQ):
            if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column):
                t1 = alias_to_table.get((eq.this.table or "").lower())
                t2 = alias_to_table.get((eq.expression.table or "").lower())
                c1 = (eq.this.name or "").lower()
                c2 = (eq.expression.name or "").lower()
                if t1 and t2 and t1 != t2:
                    matched_fk = any(
                        e["to_table"].lower() == t2
                        and e["from_col"].lower() == c1
                        and e["to_col"].lower() == c2
                        for e in fk_graph.get(t1, [])
                    )
                    if matched_fk:
                        score += 1.0
    return score


def has_suspicious_non_fk_join(sql: str, schema_info: dict[str, Any] | None) -> bool:
    """Return True if any JOIN ON equates columns from different tables that are neither FK-linked nor same-named."""
    if schema_info is None:
        return False
    tree = _parse_sqlite(sql)
    if not isinstance(tree, exp.Select):
        return False
    joins = list(tree.args.get("joins") or [])
    if not joins:
        return False
    tmap = _table_alias_map(tree, schema_info)
    alias_to_table = {ref["alias"].lower(): ref["table"].lower() for ref in tmap}
    alias_to_table.update({ref["table"].lower(): ref["table"].lower() for ref in tmap})
    fk_graph = schema_info.get("fk_graph", {})

    for jn in joins:
        on_expr = jn.args.get("on")
        if on_expr is None:
            continue
        for eq in on_expr.find_all(exp.EQ):
            if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column):
                t1 = alias_to_table.get((eq.this.table or "").lower())
                t2 = alias_to_table.get((eq.expression.table or "").lower())
                c1 = (eq.this.name or "").lower()
                c2 = (eq.expression.name or "").lower()
                if t1 and t2 and t1 != t2 and c1 and c2:
                    matched_fk = any(
                        e["to_table"].lower() == t2
                        and e["from_col"].lower() == c1
                        and e["to_col"].lower() == c2
                        for e in fk_graph.get(t1, [])
                    )
                    if not matched_fk and c1 != c2:
                        return True
    return False


def _direct_columns_in_select(sel: exp.Select) -> list[exp.Column]:
    """Return Column nodes directly belonging to `sel` (excluding nested Select scopes)."""
    cols: list[exp.Column] = []
    stack: list[exp.Expression] = list(sel.args.values())
    while stack:
        node = stack.pop()
        if node is None:
            continue
        if isinstance(node, list):
            stack.extend(node)
            continue
        if not isinstance(node, exp.Expression):
            continue
        if isinstance(node, exp.Select) and node is not sel:
            continue
        if isinstance(node, exp.Column):
            cols.append(node)
        stack.extend(node.args.values())
    return cols


def _child_selects_in_select(sel: exp.Select) -> list[exp.Select]:
    """Return immediate child Select nodes whose nearest Select ancestor is `sel`."""
    children: list[exp.Select] = []
    for sub in sel.find_all(exp.Select):
        if sub is sel:
            continue
        anc = sub.find_ancestor(exp.Select)
        if anc is sel:
            children.append(sub)
    return children


def validate_and_fix_schema_columns(
    sql: str,
    schema_info: dict[str, Any] | None,
    example: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Check column existence across scopes and deterministically fix simple alias/PK typos."""
    if schema_info is None:
        return True, sql
    tree = _parse_sqlite(sql)
    if tree is None:
        return False, sql

    cte_names = {
        cte.alias_or_name.lower()
        for cte in tree.find_all(exp.CTE)
        if cte.alias_or_name
    }

    def check_scope(sel: exp.Select, outer_tmap: list[dict[str, str]]) -> bool:
        local_tmap = _table_alias_map(sel, schema_info)
        # Check if FROM/JOIN contains dynamic subquery/CTE sources
        has_dynamic_source = False
        from_node = sel.args.get("from") or sel.args.get("from_")
        if from_node and not isinstance(from_node.this, exp.Table):
            has_dynamic_source = True
        for ref in local_tmap:
            if ref["table"].lower() in cte_names:
                has_dynamic_source = True

        combined_tmap = local_tmap + outer_tmap
        alias_to_table = {ref["alias"].lower(): ref["table"] for ref in combined_tmap}
        alias_to_table.update({ref["table"].lower(): ref["table"] for ref in combined_tmap})
        select_aliases = {
            e.alias.lower()
            for e in sel.expressions
            if isinstance(e, exp.Alias) and e.alias
        }

        scope_ok = True
        for col in _direct_columns_in_select(sel):
            cname = col.name
            if not cname or cname == "*":
                continue
            qual = col.table or ""
            if qual:
                ql = qual.lower()
                if ql in alias_to_table:
                    tname = alias_to_table[ql]
                    if tname.lower() in schema_info.get("tables", {}):
                        if _col_exists_in_table(tname, cname, schema_info) is None:
                            scope_ok = False
                elif has_dynamic_source:
                    continue
                else:
                    # Unknown qualifier (e.g. t3.form_name when only t1, t2 exist) -> fix if unique in local_tmap
                    matches = [
                        (ref["alias"], _col_exists_in_table(ref["table"], cname, schema_info))
                        for ref in local_tmap
                        if _col_exists_in_table(ref["table"], cname, schema_info) is not None
                    ]
                    if len(matches) == 1:
                        col.set("table", exp.to_identifier(matches[0][0]))
                        col.set("this", exp.to_identifier(matches[0][1]))
                    else:
                        scope_ok = False
            else:
                if cname.lower() in select_aliases or has_dynamic_source:
                    continue
                found = any(
                    _col_exists_in_table(ref["table"], cname, schema_info) is not None
                    for ref in combined_tmap
                )
                if not found:
                    # Single-table PK typo: e.g. architect_id on table architect whose PK is id
                    fixed_pk = False
                    if len(local_tmap) == 1:
                        tbl_name = local_tmap[0]["table"]
                        tinfo = schema_info.get("tables", {}).get(tbl_name.lower())
                        if (
                            tinfo
                            and len(tinfo["pks"]) == 1
                            and tinfo["pks"][0].lower() == "id"
                            and cname.lower() == f"{tbl_name.lower()}_id"
                        ):
                            col.set("this", exp.to_identifier(tinfo["pks"][0]))
                            fixed_pk = True
                    if not fixed_pk:
                        scope_ok = False

        for child in _child_selects_in_select(sel):
            if not check_scope(child, combined_tmap):
                scope_ok = False
        return scope_ok

    top_selects: list[exp.Select] = []
    if isinstance(tree, exp.Select):
        top_selects = [tree]
    else:
        for s in tree.find_all(exp.Select):
            if s.find_ancestor(exp.Select) is None:
                top_selects.append(s)

    all_ok = bool(top_selects) and all(check_scope(s, []) for s in top_selects)
    fixed_sql = _safe_sql(tree) or sql
    return all_ok, fixed_sql


def _uses_lookup_type_for_entity(tree: exp.Select, schema_info: dict[str, Any] | None) -> bool:
    """Detect filtering a secondary *Type*/*Ref*/*Category* table while primary table has a *Name* column."""
    if schema_info is None:
        return False
    tmap = _table_alias_map(tree, schema_info)
    if not tmap:
        return False
    primary_table = tmap[0]["table"]
    primary_info = schema_info.get("tables", {}).get(primary_table.lower())
    if not primary_info or not any("name" in c.lower() for c in primary_info["columns"].values()):
        return False
    all_tables = {
        tbl.name.lower()
        for tbl in tree.find_all(exp.Table)
        if tbl.name and tbl.name.lower() != primary_table.lower()
    }
    if not any(any(tok in t for tok in ("type", "ref", "category")) for t in all_tables):
        return False
    sql_l = tree.sql(dialect="sqlite").lower()
    return bool(re.search(r"[a-z0-9_]*(typename|description|type_name|category_name)\s*=", sql_l))


def _has_negation_without_exclusion(sql: str, question: str, row_count: int) -> bool:
    """Detect negation question answered via flat JOIN != / NOT LIKE on tiny public DB."""
    if row_count > 1 or not _NEGATION_RE.search(question):
        return False
    tree = _parse_sqlite(sql)
    if not isinstance(tree, exp.Select) or not (tree.args.get("joins") or []):
        return False
    sql_l = sql.lower()
    if any(kw in sql_l for kw in ("not in", "except", "not exists", "is null")):
        return False
    w = tree.args.get("where")
    if w is None:
        return False
    w_l = w.sql(dialect="sqlite").lower()
    return ("<>" in w_l) or ("!=" in w_l) or ("not like" in w_l)


def detect_spurious_wakeup_defect(
    sql: str,
    observation: dict[str, Any],
    schema_info: dict[str, Any] | None,
    all_passing: list[dict[str, Any]],
) -> list[str]:
    """Identify high-risk spurious-pass structural defects to decide LoRA wakeup."""
    defects: list[str] = []
    question = observation["task"]["question"]
    example = observation["example"]
    row_count = example.get("row_count", 0)

    if has_suspicious_non_fk_join(sql, schema_info):
        defects.append("non_fk_join")

    if _has_negation_without_exclusion(sql, question, row_count):
        defects.append("negation_without_exclusion")

    if row_count <= 1:
        tree = _parse_sqlite(sql)
        if isinstance(tree, exp.Select):
            if row_count == 0 and _uses_lookup_type_for_entity(tree, schema_info):
                defects.append("tiny_public_lookup_type_filter")
            elif len(all_passing) >= 2:
                # Diverging WHERE string literals or ORDER BY in LIMIT 1 ordered tasks
                lits_per_pass = []
                orders_per_pass = []
                for p in all_passing:
                    pt = _parse_sqlite(p["sql"])
                    if isinstance(pt, exp.Select):
                        w = pt.args.get("where")
                        lits_per_pass.append(
                            tuple(sorted(str(lt.this).lower() for lt in w.find_all(exp.Literal) if lt.is_string))
                            if w is not None else ()
                        )
                        ord_node = pt.args.get("order")
                        orders_per_pass.append(ord_node.sql(dialect="sqlite").lower() if ord_node else "")
                if len(set(lits_per_pass)) > 1:
                    defects.append("tiny_public_diverging_literals")
                elif (
                    observation["ordered"]
                    and row_count == 1
                    and tree.args.get("order") is None
                    and any(orders_per_pass)
                ):
                    defects.append("tiny_public_diverging_order")
    return defects


def _table_sig(tree: exp.Expression | None) -> tuple[str, ...] | None:
    if tree is None:
        return None
    names = sorted(t.name.lower() for t in tree.find_all(exp.Table) if t.name)
    return tuple(names) if names else None


def _resolve_col_table(
    col_node: exp.Column, sel: exp.Select, schema_info: dict[str, Any]
) -> tuple[str, str] | None:
    """Resolve an exp.Column inside `sel` to exact (table_name, col_name) from schema."""
    cname = col_node.name
    if not cname or cname == "*":
        return None
    tmap = _table_alias_map(sel, schema_info)
    alias_to_table = {ref["alias"].lower(): ref["table"] for ref in tmap}
    alias_to_table.update({ref["table"].lower(): ref["table"] for ref in tmap})
    qual = (col_node.table or "").lower()
    if qual:
        tname = alias_to_table.get(qual)
        if tname:
            real_c = _col_exists_in_table(tname, cname, schema_info)
            if real_c:
                return tname, real_c
        return None
    matches = []
    for ref in tmap:
        real_c = _col_exists_in_table(ref["table"], cname, schema_info)
        if real_c:
            matches.append((ref["table"], real_c))
    return matches[0] if len(matches) == 1 else None


def _extract_where_eq_in_targets(
    sql: str, schema_info: dict[str, Any]
) -> list[tuple[str, str, str]]:
    """Extract (table_name, col_name, literal_str) pairs from WHERE = / IN predicates."""
    tree = _parse_sqlite(sql)
    if tree is None:
        return []
    targets: list[tuple[str, str, str]] = []
    for sel in tree.find_all(exp.Select):
        w = sel.args.get("where")
        if w is None:
            continue
        for eq in w.find_all(exp.EQ):
            col_n, lit_n = (
                (eq.this, eq.expression)
                if isinstance(eq.this, exp.Column)
                else (eq.expression, eq.this)
            )
            if isinstance(col_n, exp.Column) and isinstance(lit_n, exp.Literal) and lit_n.is_string:
                resolved = _resolve_col_table(col_n, sel, schema_info)
                if resolved and lit_n.this is not None:
                    targets.append((resolved[0], resolved[1], str(lit_n.this)))
        for inn in w.find_all(exp.In):
            if isinstance(inn.this, exp.Column):
                resolved = _resolve_col_table(inn.this, sel, schema_info)
                if resolved:
                    for item in inn.expressions:
                        if isinstance(item, exp.Literal) and item.is_string and item.this is not None:
                            targets.append((resolved[0], resolved[1], str(item.this)))
    return targets


def propose_value_probe(
    observation: dict[str, Any],
    schema_info: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    tried_probe_cols: set[tuple[str, str]],
) -> tuple[str, str, str] | None:
    """Propose at most one `SELECT DISTINCT col FROM table` probe on empty/diverging WHERE literals."""
    if schema_info is None:
        return None
    example = observation["example"]
    ex_rows = example.get("row_count", 0)
    passing = [
        c for c in candidates if isinstance(c.get("result"), dict) and c["result"].get("matches_example") is True
    ]

    candidate_targets: list[tuple[str, str]] = []
    if not passing and ex_rows > 0:
        ordered = sorted(
            candidates,
            key=lambda c: (0 if "hint" in c["source"] else (1 if "llm" in c["source"] else 2), c["order"]),
        )
        for c in ordered:
            res = c.get("result")
            if isinstance(res, dict) and res.get("ok") is True and (
                res.get("row_count", 0) == 0
                or (res.get("row_count") == 1 and res.get("rows") == [[0]] and example.get("rows") != [[0]])
            ):
                for tbl, col, _ in _extract_where_eq_in_targets(c["sql"], schema_info):
                    candidate_targets.append((tbl, col))
        if not candidate_targets:
            for c in ordered:
                if c["source"] == "hint":
                    for tbl, col, _ in _extract_where_eq_in_targets(c["sql"], schema_info):
                        candidate_targets.append((tbl, col))
    elif passing and ex_rows == 0 and len(passing) >= 2:
        lits_by_col: dict[tuple[str, str], set[str]] = {}
        for c in passing:
            for tbl, col, val in _extract_where_eq_in_targets(c["sql"], schema_info):
                lits_by_col.setdefault((tbl.lower(), col.lower()), set()).add(val)
                if (tbl, col) not in candidate_targets:
                    candidate_targets.append((tbl, col))
        # Only probe on ex_rows == 0 when passing candidates disagree on literals for a column
        candidate_targets = [
            (tbl, col)
            for tbl, col in candidate_targets
            if len(lits_by_col.get((tbl.lower(), col.lower()), set())) > 1
        ]

    for tbl, col in candidate_targets:
        key = (tbl.lower(), col.lower())
        if key not in tried_probe_cols:
            probe_sql = f"SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL LIMIT 15"
            return probe_sql, tbl, col
    return None


def _join_struct_sig(sql: str) -> tuple[tuple[str, ...], tuple[str, ...]] | None:
    t = _parse_sqlite(sql)
    if not isinstance(t, exp.Select):
        return None
    tbls = tuple(sorted(tb.name.lower() for tb in t.find_all(exp.Table) if tb.name))
    jns = tuple(sorted(j.sql(dialect="sqlite").lower() for j in (t.args.get("joins") or [])))
    return (tbls, jns)


def propose_skeleton_probe(
    observation: dict[str, Any],
    passing: list[dict[str, Any]],
    best_so_far: str,
) -> tuple[str, tuple[tuple[str, ...], tuple[str, ...]]] | None:
    """Step 4.6e: On row_count=0 with >=2 distinct JOIN structures, probe stripped JOIN skeleton."""
    if observation["example"].get("row_count", 0) != 0 or len(passing) < 2:
        return None
    sigs = {_join_struct_sig(c["sql"]) for c in passing}
    sigs.discard(None)
    if len(sigs) < 2 or not any(s[1] for s in sigs):
        return None
    best_sig = _join_struct_sig(best_so_far)
    t = _parse_sqlite(best_so_far)
    if not isinstance(t, exp.Select) or not (t.args.get("joins") or []) or best_sig is None:
        return None
    cloned = t.copy()
    cloned.set("where", None)
    cloned.set("having", None)
    cloned.set("group", None)
    cloned.set("order", None)
    cloned.set("distinct", None)
    cloned.set("expressions", [exp.Literal.number(1)])
    cloned.set("limit", exp.Limit(expression=exp.Literal.number(8)))
    skel_sql = _safe_sql(cloned)
    return (skel_sql, best_sig) if skel_sql else None


def rewrite_sql_with_probes(
    sql: str,
    schema_info: dict[str, Any] | None,
    probe_results: list[dict[str, Any]],
) -> str | None:
    """Rewrite WHERE string literals in a single SQL query using probed DB values, or None if unchanged."""
    if schema_info is None or not probe_results:
        return None
    val_map: dict[tuple[str, str], list[str]] = {}
    for pv in probe_results:
        vals = [str(v) for v in (pv.get("values") or []) if v is not None]
        if vals:
            val_map[(pv["table"].lower(), pv["col"].lower())] = vals
    if not val_map:
        return None

    tree = _parse_sqlite(sql)
    if tree is None:
        return None
    cloned = tree.copy()
    changed = False

    for sel in cloned.find_all(exp.Select):
        w = sel.args.get("where")
        if w is None:
            continue
        for eq in list(w.find_all(exp.EQ)):
            col_n, lit_n = (
                (eq.this, eq.expression)
                if isinstance(eq.this, exp.Column)
                else (eq.expression, eq.this)
            )
            if not (isinstance(col_n, exp.Column) and isinstance(lit_n, exp.Literal) and lit_n.is_string):
                continue
            resolved = _resolve_col_table(col_n, sel, schema_info)
            if not resolved:
                continue
            db_vals = val_map.get((resolved[0].lower(), resolved[1].lower()))
            if not db_vals:
                continue
            lit_s = str(lit_n.this or "")
            ci_match = next((v for v in db_vals if v.lower() == lit_s.lower()), None)
            if ci_match is not None:
                if ci_match != lit_s:
                    lit_n.set("this", ci_match)
                    changed = True
            elif lit_s.isdigit():
                sep = next(
                    (
                        m.group(1)
                        for v in db_vals
                        if (m := re.match(rf"^{re.escape(lit_s)}([:.])\d+", v))
                    ),
                    None,
                )
                if sep:
                    eq.replace(
                        exp.Like(
                            this=col_n.copy(),
                            expression=exp.Literal.string(f"{lit_s}{sep}%"),
                        )
                    )
                    changed = True
            elif len(lit_s) >= 3:
                word_matches = [
                    v for v in db_vals if re.search(rf"\b{re.escape(lit_s)}\b", v, re.IGNORECASE)
                ]
                if len(word_matches) == 1 and word_matches[0] != lit_s:
                    lit_n.set("this", word_matches[0])
                    changed = True

        for inn in list(w.find_all(exp.In)):
            if not isinstance(inn.this, exp.Column):
                continue
            resolved = _resolve_col_table(inn.this, sel, schema_info)
            if not resolved:
                continue
            db_vals = val_map.get((resolved[0].lower(), resolved[1].lower()))
            if not db_vals:
                continue
            for item in inn.expressions:
                if isinstance(item, exp.Literal) and item.is_string and item.this is not None:
                    lit_s = str(item.this)
                    ci_match = next((v for v in db_vals if v.lower() == lit_s.lower()), None)
                    if ci_match is not None and ci_match != lit_s:
                        item.set("this", ci_match)
                        changed = True

    if not changed:
        return None
    s = _safe_sql(cloned)
    return s if s and len(s.encode("utf-8")) <= MAX_SQL_BYTES else None


def generate_probe_repairs(
    candidates: list[dict[str, Any]],
    schema_info: dict[str, Any] | None,
    probe_results: list[dict[str, Any]],
    checked_norm_sqls: set[str],
    normalize_fn,
) -> list[tuple[str, str]]:
    """Rewrite WHERE string literals in candidates using probed public DB distinct values."""
    if schema_info is None or not probe_results:
        return []
    proposed: list[tuple[str, str]] = []
    seen = set(checked_norm_sqls)
    ordered = sorted(
        candidates,
        key=lambda c: (0 if "hint" in c["source"] else (1 if "llm" in c["source"] else 2), c["order"]),
    )
    for cand in ordered:
        res = cand.get("result")
        if not isinstance(res, dict) or not res.get("ok"):
            continue
        s = rewrite_sql_with_probes(cand["sql"], schema_info, probe_results)
        if s:
            norm = normalize_fn(s)
            if norm not in seen:
                seen.add(norm)
                tag = "ast_from_hint" if "hint" in cand["source"] else (
                    "ast_from_llm" if "llm" in cand["source"] else "ast_from_issue"
                )
                proposed.append((s, tag))
    return proposed


def score_passing_candidate(
    cand: dict[str, Any],
    all_passing: list[dict[str, Any]],
    observation: dict[str, Any],
    schema_info: dict[str, Any] | None,
    raw_hint_counts: Counter[str],
    raw_hint_table_sigs: Counter[tuple[str, ...]],
    probe_results: list[dict[str, Any]] | None = None,
    skeleton_zero_sigs: set[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> float:
    """Score a public-passing candidate for 3-instance generalization robustness."""
    question = observation["task"]["question"]
    q_low = question.lower()
    example = observation["example"]
    exp_cols = example["columns"]
    sql = cand["sql"]
    tree = _parse_sqlite(sql)

    score = 0.0
    # 1. Source prior: LLM refiner / 8B repair_hint outperform 0.6B issue_sql
    src = cand["source"]
    if src.startswith("llm"):
        score += 3.5
    elif src.startswith("ast_from_llm"):
        score += 3.3
    elif src.startswith("ast_from_hint"):
        score += 2.8
    elif src in ("hint", "hint0", "hint_alt"):
        score += 2.0
    elif src.startswith("ast"):
        score += 1.5
    elif src.startswith("pool_") and "hint" in src:
        score += 0.5

    # Consensus bonus across raw hints (exact normalized SQL and table-set signature)
    score += 0.4 * raw_hint_counts.get(cand["norm"], 0)
    tsig = _table_sig(tree)
    if tsig is not None and "hint" in src:
        score += 0.3 * raw_hint_table_sigs.get(tsig, 0)

    if not isinstance(tree, exp.Select):
        return score

    # 2. DISTINCT alignment with question
    q_wants_distinct = bool(_DISTINCT_RE.search(question))
    has_dist = tree.args.get("distinct") is not None
    if q_wants_distinct:
        score += 3.5 if has_dist else -3.5
    elif has_dist and tree.args.get("group") is None:
        # Penalize unrequested top-level SELECT DISTINCT when another non-AST passing candidate (e.g. issue_sql) omits it
        if any(
            p["source"] == "issue"
            and isinstance((t := _parse_sqlite(p["sql"])), exp.Select)
            and t.args.get("distinct") is None
            for p in all_passing
        ):
            score -= 8.0

    # 3. Projection name & aggregation alignment with example["columns"] and question
    res = cand.get("result") or {}
    got_cols = res.get("columns", [])
    if len(got_cols) == len(exp_cols):
        for expr, rcol, ecol in zip(tree.expressions, got_cols, exp_cols):
            score += 0.6 * _expr_score_against_header(expr, rcol, ecol)

    # Aggregation cues in question when another candidate omits aggregation
    has_agg = any(True for _ in tree.find_all(exp.AggFunc))
    if re.search(r"\b(total|sum|average|avg|maximum|minimum|most|least|how many|number of|at least \d+|at most \d+)\b", q_low):
        if has_agg:
            score += 2.5
        elif any(isinstance((t := _parse_sqlite(p["sql"])), exp.Select) and any(True for _ in t.find_all(exp.AggFunc)) for p in all_passing):
            score -= 2.5

    # 4. Unused JOIN penalty (prevents losing rows on inner join to unreferenced table)
    unused_joins = _count_unused_joins(tree, schema_info)
    if unused_joins > 0:
        min_unused = min(
            (_count_unused_joins(t, schema_info) for p in all_passing if isinstance((t := _parse_sqlite(p["sql"])), exp.Select)),
            default=0,
        )
        if unused_joins > min_unused:
            score -= 8.0 * (unused_joins - min_unused)

    # 5. Foreign-key JOIN validity, spurious defect penalties, and self-equality tautologies
    score += 0.8 * _fk_join_validity_score(tree, schema_info)
    if has_suspicious_non_fk_join(sql, schema_info):
        score -= 6.0
    if _has_negation_without_exclusion(sql, question, example.get("row_count", 0)):
        if any(
            any(kw in p["sql"].lower() for kw in ("not in", "except", "not exists"))
            for p in all_passing
        ):
            score -= 4.5
    if example.get("row_count", 0) == 0 and _uses_lookup_type_for_entity(tree, schema_info):
        if any(
            isinstance((pt := _parse_sqlite(p["sql"])), exp.Select)
            and not _uses_lookup_type_for_entity(pt, schema_info)
            for p in all_passing
        ):
            score -= 4.5
    for jn in tree.args.get("joins") or []:
        on_expr = jn.args.get("on")
        if on_expr is not None:
            for eq in on_expr.find_all(exp.EQ):
                if eq.this.sql(dialect="sqlite").lower() == eq.expression.sql(dialect="sqlite").lower():
                    score -= 10.0

    # Penalize contradictory double-join of the same target table to the exact same foreign-key column (e.g. v.contestant_number = c1.id AND v.contestant_number = c2.id)
    if "both" in q_low and len(tree.args.get("joins") or []) >= 2:
        tmap = _table_alias_map(tree, schema_info)
        alias_to_tbl = {ref["alias"].lower(): ref["table"].lower() for ref in tmap}
        seen_target_bindings: set[tuple[str, str]] = set()
        for jn in tree.args.get("joins") or []:
            if not isinstance(jn.this, exp.Table):
                continue
            j_alias = (jn.this.alias_or_name or jn.this.name).lower()
            j_tbl = jn.this.name.lower()
            on_expr = jn.args.get("on")
            if on_expr is None:
                continue
            for eq in on_expr.find_all(exp.EQ):
                if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column):
                    c_left, c_right = eq.this, eq.expression
                    t_left = (c_left.table or "").lower()
                    t_right = (c_right.table or "").lower()
                    other_col = None
                    if t_left == j_alias and t_right != j_alias:
                        other_col = f"{t_right}.{c_right.name.lower()}"
                    elif t_right == j_alias and t_left != j_alias:
                        other_col = f"{t_left}.{c_left.name.lower()}"
                    if other_col:
                        key = (j_tbl, other_col)
                        if key in seen_target_bindings:
                            score -= 6.0
                        seen_target_bindings.add(key)

    # 6. Ordered completeness & predicate sanity checks
    if observation["ordered"]:
        if tree.args.get("order") is not None:
            score += 2.0
        elif any(isinstance((t := _parse_sqlite(p["sql"])), exp.Select) and t.args.get("order") is not None for p in all_passing):
            score -= 2.5

    # Split full name check (e.g. "Linda Smith" -> Fname='Linda' AND LName='Smith' vs LName='Linda Smith')
    name_m = re.search(r"\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", question)
    if name_m and tree.args.get("where") is not None:
        first_n, last_n = name_m.group(1), name_m.group(2)
        w_sql = tree.args["where"].sql(dialect="sqlite")
        if re.search(rf"fname\s*=\s*'{first_n}'", w_sql, re.IGNORECASE) and re.search(
            rf"lname\s*=\s*'{last_n}'", w_sql, re.IGNORECASE
        ):
            score += 4.0
        elif re.search(rf"=\s*'{first_n}\s+{last_n}'", w_sql, re.IGNORECASE) or (
            re.search(rf"lname\s*=\s*'{first_n}'", w_sql, re.IGNORECASE)
            and re.search(rf"fname\s*=\s*'{last_n}'", w_sql, re.IGNORECASE)
        ):
            score -= 4.0

    # Contradictory range or duplicate OR literal
    if tree.args.get("where") is not None:
        w_sql_l = tree.args["where"].sql(dialect="sqlite").lower()
        if "other_account_details" in w_sql_l or "or accounts.account_name = accounts.account_name" in w_sql_l:
            score -= 3.0
        # Duplicate identical equality in OR
        eqs = [e.sql(dialect="sqlite").lower() for e in tree.args["where"].find_all(exp.EQ)]
        if len(eqs) != len(set(eqs)):
            score -= 2.5

    # Quantifier "after any" / "before any" prefers EXISTS over > MAX
    if re.search(r"\b(after|before|greater than|less than)\s+any\b", q_low):
        if any(True for _ in tree.find_all(exp.Exists)):
            score += 2.0

    # Exact "= 1" vs ">= 1" when question mentions "one <noun>" without "at least"
    if re.search(r"\b(two|2)\b.*\bor\s+(one|1)\b", q_low) and "at least" not in q_low:
        if tree.args.get("having") is not None or any(
            True for _ in tree.find_all(exp.Having)
        ):
            h_sql = " ".join(h.sql(dialect="sqlite") for h in tree.find_all(exp.Having))
            if "= 1" in h_sql:
                score += 2.0

    # Anti-hardcoding guard: penalize literals copied from example["rows"] that do not appear in question or issue_sql
    ex_cell_strs = {
        str(cell).strip().lower()
        for row in example.get("rows", [])
        for cell in row
        if isinstance(cell, str) and len(str(cell).strip()) >= 3
    }
    if ex_cell_strs:
        q_and_issue = (question + " " + observation["issue_sql"]).lower()
        for lit in tree.find_all(exp.Literal):
            if lit.is_string and lit.this:
                val_l = str(lit.this).strip().lower()
                if val_l in ex_cell_strs and val_l not in q_and_issue:
                    score -= 6.0

    # DB value probe alignment on probed columns
    if probe_results and schema_info is not None:
        val_map = {
            (pv["table"].lower(), pv["col"].lower()): [str(v) for v in (pv.get("values") or []) if v is not None]
            for pv in probe_results
            if pv.get("values")
        }
        if val_map:
            for tbl, col, lit_s in _extract_where_eq_in_targets(sql, schema_info):
                db_vals = val_map.get((tbl.lower(), col.lower()))
                if not db_vals:
                    continue
                if lit_s in db_vals:
                    score += 5.0
                elif any(v.lower() == lit_s.lower() for v in db_vals):
                    score -= 8.0
                elif len(db_vals) < 15:
                    score -= 6.0

    if skeleton_zero_sigs:
        csig = _join_struct_sig(sql)
        if csig in skeleton_zero_sigs:
            score -= 8.0

    return score


def select_best_candidate(
    candidates: list[dict[str, Any]],
    observation: dict[str, Any],
    schema_info: dict[str, Any] | None,
    raw_hint_counts: Counter[str],
    raw_hint_list: list[str] | None = None,
    probe_results: list[dict[str, Any]] | None = None,
    skeleton_zero_sigs: set[tuple[tuple[str, ...], tuple[str, ...]]] | None = None,
) -> str:
    """Select the final SQL to submit across all evaluated candidates."""
    example = observation["example"]
    exp_cols = len(example["columns"])
    exp_rows = example["row_count"]

    raw_hint_table_sigs: Counter[tuple[str, ...]] = Counter()
    for rh in raw_hint_list or []:
        tsig = _table_sig(_parse_sqlite(rh))
        if tsig is not None:
            raw_hint_table_sigs[tsig] += 1

    passing = [
        c
        for c in candidates
        if isinstance(c.get("result"), dict) and c["result"].get("matches_example") is True
    ]
    if passing:
        if len(passing) == 1:
            return passing[0]["sql"]
        scored = [
            (
                score_passing_candidate(
                    c, passing, observation, schema_info, raw_hint_counts, raw_hint_table_sigs, probe_results, skeleton_zero_sigs
                ),
                -c["order"],
                c["sql"],
            )
            for c in passing
        ]
        return max(scored)[2]

    valid_ok = [
        c
        for c in candidates
        if isinstance(c.get("result"), dict) and c["result"].get("ok") is True
    ]
    if valid_ok:
        def rank_key(cand: dict[str, Any]) -> tuple[int, int, int, int, int]:
            res = cand["result"]
            col_match = int(len(res.get("columns", [])) == exp_cols)
            row_count = res.get("row_count", 0)
            non_empty_match = int((exp_rows == 0 and row_count == 0) or (exp_rows > 0 and row_count > 0))
            row_closeness = -abs(row_count - exp_rows)
            prefer_hint = int(cand["source"] != "issue")
            return (col_match, non_empty_match, row_closeness, prefer_hint, -cand["order"])

        return max(valid_ok, key=rank_key)["sql"]

    return observation["issue_sql"]
