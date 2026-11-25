"""Step 4.7c: Vector2 feature extractor (67 dims including 13 repair-diff features) & linear rankers."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

from sqlglot import exp

from ast_repair import (
    _DISTINCT_RE,
    _count_unused_joins,
    _expr_score_against_header,
    _parse_sqlite,
    has_suspicious_non_fk_join,
)

_WEIGHTS_PATH = Path(__file__).resolve().parent / "ranker_weights.json"
_STOP_LIKE_RE = re.compile(r"\b(the|a|an)\b", re.IGNORECASE)
_NEG_Q_RE = re.compile(r"\b(never|not|except|without|no)\b", re.IGNORECASE)


def _ast_stats(sql: str, schema_info: dict[str, Any] | None) -> dict[str, Any]:
    tree = _parse_sqlite(sql)
    if tree is None:
        return {"valid": False}
    tbls = [t.name.lower() for t in tree.find_all(exp.Table) if t.name]
    joins = list(tree.find_all(exp.Join))
    left_joins = sum(1 for j in joins if (j.side or "").lower() == "left")
    subqs = max(0, sum(1 for _ in tree.find_all(exp.Select)) - 1)
    is_sel = isinstance(tree, exp.Select)
    sel_cols = len(tree.expressions) if is_sel else (
        len(tree.left.expressions) if isinstance(tree, exp.SetOperation) and isinstance(tree.left, exp.Select) else 0
    )
    where_nodes = list(tree.find_all(exp.Where))
    n_eq = sum(sum(1 for _ in w.find_all(exp.EQ)) for w in where_nodes)
    n_and = sum(sum(1 for _ in w.find_all(exp.And)) for w in where_nodes)
    n_or = sum(sum(1 for _ in w.find_all(exp.Or)) for w in where_nodes)
    n_like = sum(sum(1 for _ in w.find_all(exp.Like)) for w in where_nodes)
    n_neq = sum(sum(1 for _ in w.find_all(exp.NEQ)) for w in where_nodes)
    n_in = sum(sum(1 for _ in w.find_all(exp.In)) for w in where_nodes)
    n_agg = sum(1 for _ in tree.find_all(exp.AggFunc))
    n_unused = _count_unused_joins(tree, schema_info) if is_sel else 0
    non_fk = int(has_suspicious_non_fk_join(sql, schema_info)) if is_sel else 0
    sql_l = sql.lower()
    has_self_eq = 0
    for jn in joins:
        on_e = jn.args.get("on")
        if on_e is not None:
            for eq in on_e.find_all(exp.EQ):
                if eq.this.sql(dialect="sqlite").lower() == eq.expression.sql(dialect="sqlite").lower():
                    has_self_eq = 1
    return {
        "valid": True,
        "tree": tree,
        "is_sel": is_sel,
        "tbls": tbls,
        "n_tbls": len(tbls),
        "n_joins": len(joins),
        "n_left_joins": left_joins,
        "n_subqs": subqs,
        "non_fk": non_fk,
        "n_unused": n_unused,
        "has_distinct": int(is_sel and tree.args.get("distinct") is not None),
        "has_group": int(any(True for _ in tree.find_all(exp.Group))),
        "has_having": int(any(True for _ in tree.find_all(exp.Having))),
        "has_order": int(any(True for _ in tree.find_all(exp.Order))),
        "has_limit": int(any(True for _ in tree.find_all(exp.Limit))),
        "has_except": int(" except " in sql_l),
        "has_intersect": int(" intersect " in sql_l),
        "has_union": int(" union " in sql_l),
        "has_not_in": int("not in" in sql_l or "not " in sql_l and " in (" in sql_l),
        "has_not_exists": int("not exists" in sql_l),
        "has_is_null": int("is null" in sql_l),
        "has_like": int(n_like > 0),
        "has_neq": int(n_neq > 0),
        "n_eq": n_eq,
        "n_and": n_and,
        "n_or": n_or,
        "n_in": n_in,
        "n_agg": n_agg,
        "sel_cols": sel_cols,
        "has_self_eq": has_self_eq,
        "has_sq": int("'" in sql),
    }


def extract_vector2_features(
    sql: str,
    issue_sql: str,
    question: str,
    ordered: bool,
    example: dict[str, Any],
    schema_info: dict[str, Any] | None,
    source_tag: str = "",
) -> list[float]:
    """Extract 67-dim vector2 features: 14 intent + 28 AST + 12 schema/overlap + 13 repair-diff."""
    q_l = question.lower()
    exp_cols = example.get("columns") or []
    exp_col_cnt = len(exp_cols)

    # 1. 14-dim Question Intent Flags
    intent = [
        float(bool(_DISTINCT_RE.search(q_l))),
        float(bool(re.search(r"\b(how many|count|number of)\b", q_l))),
        float(bool(re.search(r"\b(average|avg|mean)\b", q_l))),
        float(bool(re.search(r"\b(sum|total)\b", q_l))),
        float(bool(re.search(r"\b(max|maximum|highest|largest|most|oldest|newest|longest)\b", q_l))),
        float(bool(re.search(r"\b(min|minimum|lowest|smallest|least|youngest|shortest)\b", q_l))),
        float(bool(re.search(r"\b(order|sort|alphabetical|ascending|descending)\b", q_l))),
        float(bool(_NEG_Q_RE.search(q_l))),
        float("both" in q_l),
        float(" or " in q_l),
        float("at least" in q_l),
        float("at most" in q_l),
        float(bool(re.search(r"\b(each|every|per)\b", q_l))),
        float(bool(ordered)),
    ]

    st = _ast_stats(sql, schema_info)
    ist = _ast_stats(issue_sql, schema_info)
    if not st["valid"]:
        return intent + [0.0] * 53

    # 2. 28-dim SQL AST Structure & Fragility Counts
    ast_feats = [
        float(st["n_tbls"]),
        float(st["n_joins"]),
        float(st["n_left_joins"]),
        float(st["non_fk"]),
        float(st["n_unused"]),
        float(st["n_subqs"]),
        float(st["has_distinct"]),
        float(st["has_group"]),
        float(st["has_having"]),
        float(st["has_order"]),
        float(st["has_limit"]),
        float(st["has_except"]),
        float(st["has_intersect"]),
        float(st["has_union"]),
        float(st["has_not_in"]),
        float(st["has_not_exists"]),
        float(st["has_is_null"]),
        float(st["has_like"]),
        float(st["has_neq"]),
        float(st["n_eq"]),
        float(st["n_and"]),
        float(st["n_or"]),
        float(st["n_agg"]),
        float(st["sel_cols"]),
        float(st["sel_cols"] == exp_col_cnt),
        min(len(sql) / 300.0, 3.0),
        float(st["has_self_eq"]),
        float(st["has_sq"]),
    ]

    # 3. 12-dim Schema/Question Overlap & Cross/Source Features
    q_words = set(re.findall(r"[a-z0-9_]+", q_l))
    sql_words = set(re.findall(r"[a-z0-9_]+", sql.lower()))
    tbl_overlap = sum(1 for t in st["tbls"] if any(w in q_l for w in t.split("_") if len(w) >= 3)) / max(1, st["n_tbls"])
    col_overlap = len(q_words & sql_words) / max(1, len(q_words))
    ex_hdr_match = 0.0
    sel_node = (
        st["tree"]
        if st["is_sel"]
        else (st["tree"].left if isinstance(st["tree"], exp.SetOperation) and isinstance(st["tree"].left, exp.Select) else None)
    )
    if sel_node is not None and st["sel_cols"] == exp_col_cnt and exp_col_cnt > 0:
        ex_hdr_match = sum(
            _expr_score_against_header(e, "", hdr) for e, hdr in zip(sel_node.expressions, exp_cols)
        ) / exp_col_cnt

    src_l = source_tag.lower()
    cross_feats = [
        tbl_overlap,
        col_overlap,
        ex_hdr_match,
        float(intent[0] and st["has_distinct"]),
        float((intent[1] or intent[2] or intent[3]) and st["n_agg"] > 0),
        float(intent[13] and st["has_order"]),
        float(intent[7] and (st["has_not_in"] or st["has_except"] or st["has_not_exists"] or st["has_is_null"])),
        float(src_l in ("hint", "hint0", "hint_alt") or "hint" in src_l),
        float(src_l == "issue" or "issue" in src_l),
        float("tmpl" in src_l or "ast_from_hint" in src_l),
        float("simp" in src_l and "simp2" not in src_l),
        float("llm" in src_l),
    ]

    # 4. 13-dim Repair-Diff Features vs issue_sql
    if not ist["valid"]:
        diff_feats = [0.0] * 13
    else:
        same_non_proj = (
            st["tbls"] == ist["tbls"]
            and st["n_joins"] == ist["n_joins"]
            and st["has_group"] == ist["has_group"]
            and st["has_order"] == ist["has_order"]
            and st["n_eq"] == ist["n_eq"]
        )
        diff_feats = [
            float(sql.strip().lower() == issue_sql.strip().lower()),
            float(same_non_proj and sql.strip().lower() != issue_sql.strip().lower()),
            float(st["has_like"] and not ist["has_like"] and st["n_eq"] < ist["n_eq"]),
            float(st["n_joins"] < ist["n_joins"]),
            float(st["n_tbls"] < ist["n_tbls"]),
            float(st["n_subqs"] < ist["n_subqs"]),
            float(ist["has_distinct"] and not st["has_distinct"]),
            float(st["n_eq"] + st["n_and"] < ist["n_eq"] + ist["n_and"]),
            float(st["has_limit"] and not ist["has_limit"]),
            float(st["has_group"] and not ist["has_group"]),
            float(st["has_except"] and not ist["has_except"]),
            float(st["n_in"] > ist["n_in"] or st["has_not_in"] and not ist["has_not_in"]),
            float(len(sql) < len(issue_sql)),
        ]

    return intent + ast_feats + cross_feats + diff_feats


def simplicity_prior(sql: str) -> float:
    """Occam simplicity prior: prefer fewer JOINs, tables, subqueries, and shorter SQL."""
    st = _ast_stats(sql, None)
    if not st["valid"]:
        return -10.0
    return -(1.0 * st["n_joins"] + 0.8 * st["n_tbls"] + 1.2 * st["n_subqs"] + 0.005 * len(sql))


def column_match_score(sql: str, got_cols: list[str], exp_cols: list[str]) -> float:
    """Execution column header alignment score in [0, 1]."""
    if not exp_cols or len(got_cols) != len(exp_cols):
        return 0.0
    tree = _parse_sqlite(sql)
    sel_node = (
        tree
        if isinstance(tree, exp.Select)
        else (tree.left if isinstance(tree, exp.SetOperation) and isinstance(tree.left, exp.Select) else None)
    )
    if sel_node is None or len(sel_node.expressions) != len(exp_cols):
        return 0.2
    tot = sum(_expr_score_against_header(e, gc, ec) for e, gc, ec in zip(sel_node.expressions, got_cols, exp_cols))
    return max(0.0, min(1.0, (tot / len(exp_cols)) / 2.0))


def peer_relative_penalty(sql: str, all_passing: list[dict[str, Any]], question: str) -> float:
    """Step 4.7c peer-relative penalties for unrequested DISTINCT and stopword-polluted LIKE."""
    pen = 0.0
    q_wants_dist = bool(_DISTINCT_RE.search(question))
    tree = _parse_sqlite(sql)
    if isinstance(tree, exp.Select):
        if not q_wants_dist and tree.args.get("distinct") is not None:
            if any(isinstance((pt := _parse_sqlite(p["sql"])), exp.Select) and pt.args.get("distinct") is None for p in all_passing):
                pen -= 3.0
        w = tree.args.get("where")
        if w is not None:
            like_lits = [str(lk.expression.this) for lk in w.find_all(exp.Like) if isinstance(lk.expression, exp.Literal) and lk.expression.is_string]
            if any(_STOP_LIKE_RE.search(lit) for lit in like_lits):
                # Check if any peer passing candidate has no stopword-polluted LIKE
                def clean_peer(ps: str) -> bool:
                    pt = _parse_sqlite(ps)
                    if not isinstance(pt, exp.Select) or pt.args.get("where") is None:
                        return True
                    plks = [str(lk.expression.this) for lk in pt.args["where"].find_all(exp.Like) if isinstance(lk.expression, exp.Literal) and lk.expression.is_string]
                    return not any(_STOP_LIKE_RE.search(lit) for lit in plks)
                if any(clean_peer(p["sql"]) for p in all_passing):
                    pen -= 3.0
    return pen


class LinearRankerBundle:
    """Offline-loaded pair of linear rankers (priority_ranker + choice_ranker)."""

    def __init__(self, weights_path: Path = _WEIGHTS_PATH) -> None:
        self.loaded = False
        self.priority: dict[str, Any] | None = None
        self.choice: dict[str, Any] | None = None
        if weights_path.is_file():
            data = json.loads(weights_path.read_text(encoding="utf-8"))
            self.priority = data.get("priority_ranker")
            self.choice = data.get("choice_ranker")
            self.loaded = bool(self.priority and self.choice)

    @staticmethod
    def _score_model(model: dict[str, Any] | None, feats: list[float]) -> float:
        if not model:
            return 0.0
        mean = model["mean"]
        std = model["std"]
        w = model["weights"]
        b = model["bias"]
        return b + sum(((x - m) / s) * wi for x, m, s, wi in zip(feats, mean, std, w))

    def score_priority(self, feats: list[float]) -> float:
        return self._score_model(self.priority, feats)

    def score_choice(self, feats: list[float]) -> float:
        return self._score_model(self.choice, feats)
