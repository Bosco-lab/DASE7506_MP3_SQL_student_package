"""Step 4.7a: 7-layer broad static AST candidate pool (pool_cap=80) with SQLGlot canonical deduplication."""
from __future__ import annotations

import re
from typing import Any

import sqlglot
from sqlglot import exp

from ast_repair import (
    MAX_SQL_BYTES,
    _col_exists_in_table,
    _extract_col_tokens,
    _parse_sqlite,
    _resolve_col_table,
    _safe_sql,
    _table_alias_map,
)

_NEG_Q_RE = re.compile(r"\b(never|not|except|without|no)\b", re.IGNORECASE)


def canonical_sql(sql: str) -> str:
    """Return a canonical normalized SQLite string for strict deduplication."""
    tree = _parse_sqlite(sql)
    if tree is not None:
        s = _safe_sql(tree)
        if s:
            return re.sub(r"\s+", " ", s.strip()).rstrip(";").strip().lower()
    return re.sub(r"\s+", " ", (sql or "").strip()).rstrip(";").strip().lower()


def _is_valid_pool_sql(sql: str, schema_info: dict[str, Any] | None) -> bool:
    if not sql or len(sql.encode("utf-8")) > MAX_SQL_BYTES:
        return False
    tree = _parse_sqlite(sql)
    if not isinstance(tree, (exp.Select, exp.SetOperation)):
        return False
    if schema_info is not None:
        allowed = set(schema_info.get("tables", {}).keys())
        cte_names = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE) if c.alias_or_name}
        for tbl in tree.find_all(exp.Table):
            if tbl.name and tbl.name.lower() not in allowed and tbl.name.lower() not in cte_names:
                return False
    return True


def _split_conjuncts(node: exp.Expression) -> list[exp.Expression]:
    if isinstance(node, exp.And):
        return _split_conjuncts(node.this) + _split_conjuncts(node.expression)
    if isinstance(node, exp.Paren):
        return _split_conjuncts(node.this)
    return [node]


def _join_conjuncts(nodes: list[exp.Expression]) -> exp.Expression | None:
    if not nodes:
        return None
    cur = nodes[0].copy()
    for n in nodes[1:]:
        cur = exp.And(this=cur, expression=n.copy())
    return cur


def _column_guided_projections(
    tree: exp.Select, example: dict[str, Any], schema_info: dict[str, Any] | None
) -> list[str]:
    out: list[str] = []
    exp_cols = example.get("columns") or []
    if not exp_cols:
        return out

    exprs = list(tree.expressions or [])
    # 1. Reorder expressions if same length and distinct header tokens match
    if len(exprs) == len(exp_cols) and len(exp_cols) > 1:
        used: set[int] = set()
        reordered: list[exp.Expression] = []
        ok = True
        for hdr in exp_cols:
            t_agg, t_toks = _extract_col_tokens(hdr)
            best_i = -1
            best_s = -1
            for i, e in enumerate(exprs):
                if i in used:
                    continue
                es = e.sql(dialect="sqlite").lower()
                score = sum(1 for tok in t_toks if tok in es)
                if t_agg and t_agg in es:
                    score += 2
                if score > best_s:
                    best_s = score
                    best_i = i
            if best_i >= 0 and best_s > 0:
                used.add(best_i)
                reordered.append(exprs[best_i].copy())
            else:
                ok = False
                break
        if ok and len(reordered) == len(exprs):
            cloned = tree.copy()
            cloned.set("expressions", reordered)
            s = _safe_sql(cloned)
            if s:
                out.append(s)

    # 2. Trim extra projection columns if more columns than expected
    if len(exprs) > len(exp_cols):
        cloned = tree.copy()
        cloned.set("expressions", [e.copy() for e in exprs[: len(exp_cols)]])
        s = _safe_sql(cloned)
        if s:
            out.append(s)

    # 3. Replace single aggregate / add missing column from active tables when fewer columns
    if len(exprs) < len(exp_cols) and schema_info is not None:
        tmap = _table_alias_map(tree, schema_info)
        added: list[exp.Expression] = [e.copy() for e in exprs]
        for hdr in exp_cols[len(exprs) :]:
            t_agg, t_toks = _extract_col_tokens(hdr)
            if t_agg == "count":
                added.append(exp.Count(this=exp.Star()))
            else:
                matched_col = None
                for ref in tmap:
                    tinfo = schema_info["tables"].get(ref["table"].lower())
                    if not tinfo:
                        continue
                    for cname in tinfo["columns"].values():
                        if any(tok == cname.lower() or tok in cname.lower() for tok in t_toks):
                            matched_col = exp.Column(this=exp.to_identifier(cname), table=exp.to_identifier(ref["alias"]))
                            break
                    if matched_col is not None:
                        break
                if matched_col is not None:
                    added.append(matched_col)
        if len(added) == len(exp_cols):
            cloned = tree.copy()
            cloned.set("expressions", added)
            s = _safe_sql(cloned)
            if s:
                out.append(s)
    return out


def _unqualify_and_qualify_variants(
    tree: exp.Select, schema_info: dict[str, Any] | None
) -> list[str]:
    out: list[str] = []
    if schema_info is None:
        return out
    tmap = _table_alias_map(tree, schema_info)

    # Unqualify when single table
    if len(tmap) == 1:
        cloned = tree.copy()
        changed = False
        for c in cloned.find_all(exp.Column):
            if c.table:
                c.set("table", None)
                changed = True
        if changed:
            s = _safe_sql(cloned)
            if s:
                out.append(s)

    # Qualify unqualified columns that uniquely resolve in tmap
    if tmap:
        cloned = tree.copy()
        changed = False
        for sel in cloned.find_all(exp.Select):
            local_tmap = _table_alias_map(sel, schema_info)
            for c in sel.find_all(exp.Column):
                if c.find_ancestor(exp.Select) is not sel or c.table or not c.name or c.name == "*":
                    continue
                hits = [
                    (ref["alias"], _col_exists_in_table(ref["table"], c.name, schema_info))
                    for ref in local_tmap
                    if _col_exists_in_table(ref["table"], c.name, schema_info) is not None
                ]
                if len(hits) == 1:
                    c.set("table", exp.to_identifier(hits[0][0]))
                    c.set("this", exp.to_identifier(hits[0][1]))
                    changed = True
        if changed:
            s = _safe_sql(cloned)
            if s:
                out.append(s)
    return out


def _simplifications(
    tree: exp.Select, schema_info: dict[str, Any] | None, question: str = "", *, strip_percent: bool = False
) -> list[str]:
    out: list[str] = []
    joins = list(tree.args.get("joins") or [])
    q_low = (question or "").lower()

    # 0. Drop all joins at once when >=2 joins and non-join clauses only reference the base FROM table
    from_n = tree.args.get("from") or tree.args.get("from_")
    if len(joins) >= 2 and from_n is not None and isinstance(from_n.this, exp.Table):
        base_aliases = {from_n.this.name.lower(), (from_n.this.alias_or_name or from_n.this.name).lower()}
        only_base = True
        for part_name in ("expressions", "where", "group", "having", "order"):
            part = tree.args.get(part_name)
            nodes = part if isinstance(part, list) else ([part] if part is not None else [])
            for n in nodes:
                for c in n.find_all(exp.Column):
                    if c.table and c.table.lower() not in base_aliases:
                        only_base = False
                        break
        if only_base:
            cloned = tree.copy()
            cloned.set("joins", [])
            s = _safe_sql(cloned)
            if s and _is_valid_pool_sql(s, schema_info):
                out.append(s)

    # 1. Drop joins one at a time
    for idx in range(len(joins)):
        cloned = tree.copy()
        rem_joins = [j.copy() for j_i, j in enumerate(joins) if j_i != idx]
        cloned.set("joins", rem_joins)
        s = _safe_sql(cloned)
        if s and _is_valid_pool_sql(s, schema_info):
            out.append(s)

    # 2. Strip outer wrapper subquery (SELECT ... FROM (SELECT ...))
    from_n = tree.args.get("from") or tree.args.get("from_")
    if from_n and isinstance(from_n.this, exp.Subquery) and isinstance(from_n.this.this, exp.Select):
        s = _safe_sql(from_n.this.this)
        if s:
            out.append(s)

    # 3. _prune_global_aggregate: drop redundant GROUP BY / HAVING
    if tree.args.get("having") is not None:
        cloned = tree.copy()
        cloned.set("having", None)
        s = _safe_sql(cloned)
        if s:
            out.append(s)
    if tree.args.get("group") is not None and not any(True for _ in tree.find_all(exp.AggFunc)):
        cloned = tree.copy()
        cloned.set("group", None)
        s = _safe_sql(cloned)
        if s:
            out.append(s)

    # 4. Relax over-constrained WHERE conjuncts (only drop a conjunct if its literals are not anchored in question; never strip entire WHERE)
    w = tree.args.get("where")
    if w is not None:
        conjuncts = _split_conjuncts(w.this)
        if len(conjuncts) > 1:
            for drop_i, dropped_c in enumerate(conjuncts):
                anchored = False
                for lit in dropped_c.find_all(exp.Literal):
                    val_s = str(lit.this or "").strip().lower()
                    if strip_percent and isinstance(lit.parent, exp.Like):
                        val_s = val_s.strip("%")
                    if val_s and val_s in q_low:
                        anchored = True
                        break
                if anchored:
                    continue
                kept = [c for j, c in enumerate(conjuncts) if j != drop_i]
                new_w = _join_conjuncts(kept)
                if new_w is not None:
                    cloned = tree.copy()
                    cloned.set("where", exp.Where(this=new_w))
                    s = _safe_sql(cloned)
                    if s:
                        out.append(s)

    # 5. Toggle DISTINCT and LIMIT 1
    if tree.args.get("distinct") is not None:
        cloned = tree.copy()
        cloned.set("distinct", None)
        s = _safe_sql(cloned)
        if s:
            out.append(s)
    else:
        cloned = tree.copy()
        cloned.set("distinct", exp.Distinct())
        s = _safe_sql(cloned)
        if s:
            out.append(s)

    if tree.args.get("limit") is not None:
        cloned = tree.copy()
        cloned.set("limit", None)
        s = _safe_sql(cloned)
        if s:
            out.append(s)
    elif tree.args.get("order") is not None:
        cloned = tree.copy()
        cloned.set("limit", exp.Limit(expression=exp.Literal.number(1)))
        s = _safe_sql(cloned)
        if s:
            out.append(s)

    return out


def _name_variants(tree: exp.Select, schema_info: dict[str, Any] | None) -> list[str]:
    out: list[str] = []
    if schema_info is None or tree.args.get("where") is None:
        return out
    tmap = _table_alias_map(tree, schema_info)
    for ref in tmap:
        tinfo = schema_info["tables"].get(ref["table"].lower())
        if not tinfo:
            continue
        cols_l = {c.lower(): c for c in tinfo["columns"].values()}
        f_col = next((cols_l[k] for k in ("fname", "first_name") if k in cols_l), None)
        l_col = next((cols_l[k] for k in ("lname", "last_name") if k in cols_l), None)
        if not (f_col and l_col):
            continue
        cloned = tree.copy()
        changed = False
        for eq in list(cloned.args["where"].find_all(exp.EQ)):
            col_n, lit_n = (eq.this, eq.expression) if isinstance(eq.this, exp.Column) else (eq.expression, eq.this)
            if isinstance(col_n, exp.Column) and isinstance(lit_n, exp.Literal) and lit_n.is_string:
                parts = str(lit_n.this or "").strip().split()
                if len(parts) == 2 and all(p.isalpha() for p in parts):
                    qual = col_n.table
                    c1 = exp.Column(this=exp.to_identifier(f_col), table=exp.to_identifier(qual) if qual else None)
                    c2 = exp.Column(this=exp.to_identifier(l_col), table=exp.to_identifier(qual) if qual else None)
                    eq.replace(
                        exp.And(
                            this=exp.EQ(this=c1, expression=exp.Literal.string(parts[0])),
                            expression=exp.EQ(this=c2, expression=exp.Literal.string(parts[1])),
                        )
                    )
                    changed = True
        if changed:
            s = _safe_sql(cloned)
            if s:
                out.append(s)
    return out


def _split_anti_union_and_intersect(
    tree: exp.Select, question: str, schema_info: dict[str, Any] | None, *, strip_percent: bool = False
) -> list[str]:
    out: list[str] = []
    q_low = question.lower()
    joins = list(tree.args.get("joins") or [])
    w = tree.args.get("where")

    # 1. Negation flat JOIN != / <> / NOT LIKE -> EXCEPT / NOT IN
    if _NEG_Q_RE.search(q_low) and joins and w is not None and len(tree.expressions) >= 1:
        w_sql = w.sql(dialect="sqlite")
        if any(op in w_sql.lower() for op in ("<>", "!=", "not like")):
            # Build EXCEPT: base table SELECT minus JOIN with positive condition
            from_n = tree.args.get("from") or tree.args.get("from_")
            if from_n and isinstance(from_n.this, exp.Table):
                base_tbl = from_n.this.name
                base_sel = exp.Select(expressions=[e.copy() for e in tree.expressions]).from_(base_tbl)
                pos_w = w.copy()
                for neq in list(pos_w.find_all(exp.NEQ)):
                    neq.replace(exp.EQ(this=neq.this.copy(), expression=neq.expression.copy()))
                sub_sel = tree.copy()
                sub_sel.set("where", pos_w)
                sub_sel.set("distinct", None)
                s_left = _safe_sql(base_sel)
                s_right = _safe_sql(sub_sel)
                if s_left and s_right and s_left != s_right:
                    ex_sql = f"{s_left} EXCEPT {s_right}"
                    if _is_valid_pool_sql(ex_sql, schema_info):
                        out.append(ex_sql)

    # 2. Both X and Y -> IN ('X', 'Y') to INTERSECT
    if "both" in q_low and w is not None:
        for inn in w.find_all(exp.In):
            if len(inn.expressions) == 2 and isinstance(inn.this, exp.Column):
                branches = []
                for lit in inn.expressions:
                    cl = tree.copy()
                    cl.set("group", None)
                    cl.set("having", None)
                    for sub_in in cl.args["where"].find_all(exp.In):
                        sub_in.replace(exp.EQ(this=inn.this.copy(), expression=lit.copy()))
                    bs = _safe_sql(cl)
                    if bs:
                        branches.append(bs)
                if len(branches) == 2:
                    inter_sql = f"{branches[0]} INTERSECT {branches[1]}"
                    if _is_valid_pool_sql(inter_sql, schema_info):
                        out.append(inter_sql)

    # 3. FK-based NOT IN synthesis for negation questions when current SQL uses JOIN without question-anchored WHERE literals
    from_n = tree.args.get("from") or tree.args.get("from_")
    if _NEG_Q_RE.search(q_low) and joins and schema_info is not None and from_n is not None and isinstance(from_n.this, exp.Table):
        has_anchored_where_lit = False
        if w is not None:
            for lit in w.find_all(exp.Literal):
                val_s = str(lit.this or "").strip().lower()
                if strip_percent and isinstance(lit.parent, exp.Like):
                    val_s = val_s.strip("%")
                if len(val_s) >= 2 and val_s in q_low:
                    has_anchored_where_lit = True
                    break
        if not has_anchored_where_lit:
            fk_graph = schema_info.get("fk_graph", {})
            base_tbl = from_n.this.name
            if len(joins) == 1 and isinstance(joins[0].this, exp.Table):
                j_tbl = joins[0].this.name
                edges = [e for e in fk_graph.get(base_tbl.lower(), []) if e["to_table"].lower() == j_tbl.lower()]
                if len(edges) == 1:
                    b_col = edges[0]["from_col"]
                    j_col = edges[0]["to_col"]
                    clean_exprs = []
                    for e in tree.expressions:
                        ce = e.copy()
                        for c in ce.find_all(exp.Column):
                            c.set("table", None)
                        clean_exprs.append(ce)
                    sub_sel = exp.Select(expressions=[exp.Column(this=exp.to_identifier(j_col))]).from_(j_tbl)
                    outer_sel = exp.Select(expressions=clean_exprs).from_(base_tbl)
                    outer_sel.set(
                        "where",
                        exp.Where(
                            this=exp.Not(
                                this=exp.In(
                                    this=exp.Column(this=exp.to_identifier(b_col)),
                                    query=exp.Subquery(this=sub_sel),
                                )
                            )
                        ),
                    )
                    s = _safe_sql(outer_sel)
                    if s and _is_valid_pool_sql(s, schema_info):
                        out.append(s)

    return out


def _aggregate_column_variants(tree: exp.Select, schema_info: dict[str, Any] | None) -> list[str]:
    out: list[str] = []
    # 1. COUNT(col) <-> COUNT(*) <-> COUNT(DISTINCT col)
    for idx, expr in enumerate(tree.expressions):
        inner = expr.this if isinstance(expr, exp.Alias) else expr
        if isinstance(inner, exp.Count):
            cloned = tree.copy()
            new_cnt = exp.Count(this=exp.Star()) if not isinstance(inner.this, exp.Star) else None
            if new_cnt is not None:
                if isinstance(cloned.expressions[idx], exp.Alias):
                    cloned.expressions[idx].set("this", new_cnt)
                else:
                    cloned.expressions[idx] = new_cnt
                s = _safe_sql(cloned)
                if s:
                    out.append(s)

    # 2. Swap numeric columns in SUM/AVG/MAX/MIN with same-table numeric columns
    if schema_info is not None:
        num_types = ("int", "real", "num", "float", "dec", "double")
        for agg in tree.find_all(exp.AggFunc):
            if isinstance(agg, exp.Count) or not isinstance(agg.this, exp.Column):
                continue
            resolved = _resolve_col_table(agg.this, tree, schema_info)
            if not resolved:
                continue
            tname, old_col = resolved
            tinfo = schema_info["tables"].get(tname.lower())
            if not tinfo:
                continue
            for cand_col, ctype in tinfo.get("types", {}).items():
                if cand_col.lower() == old_col.lower():
                    continue
                if any(nt in ctype.lower() for nt in num_types) and "id" not in cand_col.lower():
                    cloned = tree.copy()
                    for c_agg in cloned.find_all(type(agg)):
                        if isinstance(c_agg.this, exp.Column) and c_agg.this.name.lower() == old_col.lower():
                            c_agg.this.set("this", exp.to_identifier(cand_col))
                    s = _safe_sql(cloned)
                    if s:
                        out.append(s)
    return out


def _basic_variants(tree: exp.Select, schema_info: dict[str, Any] | None) -> list[str]:
    out: list[str] = []
    # 1. Flip ASC/DESC
    if tree.args.get("order") is not None:
        cloned = tree.copy()
        for o in cloned.args["order"].find_all(exp.Ordered):
            o.set("desc", not bool(o.args.get("desc")))
        s = _safe_sql(cloned)
        if s:
            out.append(s)

    # 2. Flip > <-> >= and < <-> <=
    cloned = tree.copy()
    flipped = False
    for gt in list(cloned.find_all(exp.GT)):
        gt.replace(exp.GTE(this=gt.this.copy(), expression=gt.expression.copy()))
        flipped = True
    for lt in list(cloned.find_all(exp.LT)):
        lt.replace(exp.LTE(this=lt.this.copy(), expression=lt.expression.copy()))
        flipped = True
    if flipped:
        s = _safe_sql(cloned)
        if s:
            out.append(s)

    # 3. Fix non-FK JOIN ON using real FK edge from schema_info
    if schema_info is not None and tree.args.get("joins"):
        fk_graph = schema_info.get("fk_graph", {})
        tmap = _table_alias_map(tree, schema_info)
        alias_to_table = {ref["alias"].lower(): ref["table"].lower() for ref in tmap}
        table_to_alias = {ref["table"].lower(): ref["alias"] for ref in tmap}
        cloned = tree.copy()
        changed = False
        for jn in cloned.args.get("joins") or []:
            on_e = jn.args.get("on")
            if on_e is None:
                continue
            for eq in on_e.find_all(exp.EQ):
                if isinstance(eq.this, exp.Column) and isinstance(eq.expression, exp.Column):
                    t1 = alias_to_table.get((eq.this.table or "").lower())
                    t2 = alias_to_table.get((eq.expression.table or "").lower())
                    c1 = (eq.this.name or "").lower()
                    c2 = (eq.expression.name or "").lower()
                    if t1 and t2 and t1 != t2:
                        edges = [e for e in fk_graph.get(t1, []) if e["to_table"].lower() == t2]
                        if len(edges) == 1 and not (
                            edges[0]["from_col"].lower() == c1 and edges[0]["to_col"].lower() == c2
                        ):
                            a1 = table_to_alias.get(t1, t1)
                            a2 = table_to_alias.get(t2, t2)
                            eq.set("this", exp.Column(this=exp.to_identifier(edges[0]["from_col"]), table=exp.to_identifier(a1)))
                            eq.set("expression", exp.Column(this=exp.to_identifier(edges[0]["to_col"]), table=exp.to_identifier(a2)))
                            changed = True
        if changed:
            s = _safe_sql(cloned)
            if s:
                out.append(s)
    return out


def build_broad_ast_pool(
    observation: dict[str, Any],
    schema_info: dict[str, Any] | None,
    raw_hints: list[str],
    pool_cap: int = 80,
) -> list[dict[str, str]]:
    """Derive up to `pool_cap` canonical-deduplicated AST candidates from issue_sql and repair_hint."""
    question = observation["task"]["question"]
    example = observation["example"]
    issue_sql = observation["issue_sql"]

    seeds: list[tuple[str, str]] = []
    if raw_hints:
        seeds.append((raw_hints[0], "hint0"))
    seeds.append((issue_sql, "issue"))
    for rh in raw_hints[1:3]:
        seeds.append((rh, "hint_alt"))

    pool: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(sql: str, tag: str) -> None:
        if len(pool) >= pool_cap:
            return
        if not _is_valid_pool_sql(sql, schema_info):
            return
        cs = canonical_sql(sql)
        if cs not in seen:
            seen.add(cs)
            pool.append({"sql": sql, "canonical": cs, "source": tag})

    # Layer 0: seeds
    for s, tag in seeds:
        add(s, tag)

    # Layers 1-6: expand each seed (prioritizing hint0 then issue)
    for seed_sql, seed_tag in seeds[:2]:
        tree = _parse_sqlite(seed_sql)
        if not isinstance(tree, exp.Select):
            continue
        for s in _column_guided_projections(tree, example, schema_info):
            add(s, f"pool_proj_{seed_tag}")
        for s in _unqualify_and_qualify_variants(tree, schema_info):
            add(s, f"pool_qual_{seed_tag}")
        for s in _simplifications(tree, schema_info, question):
            add(s, f"pool_simp_{seed_tag}")
        for s in _name_variants(tree, schema_info):
            add(s, f"pool_name_{seed_tag}")
        for s in _split_anti_union_and_intersect(tree, question, schema_info):
            add(s, f"pool_tmpl_{seed_tag}")
        for s in _aggregate_column_variants(tree, schema_info):
            add(s, f"pool_agg_{seed_tag}")
        for s in _basic_variants(tree, schema_info):
            add(s, f"pool_basic_{seed_tag}")

    # Layer 7: second-order simplifications on already derived pool_qual/pool_proj/pool_tmpl
    first_wave = list(pool[:20])
    for item in first_wave:
        if len(pool) >= pool_cap:
            break
        t = _parse_sqlite(item["sql"])
        if not isinstance(t, exp.Select):
            continue
        seed_suffix = "hint0" if "hint" in item["source"] else "issue"
        for s in _simplifications(t, schema_info, question)[:4]:
            add(s, f"pool_simp2_{seed_suffix}")

    return pool[:pool_cap]
