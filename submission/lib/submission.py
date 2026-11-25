"""DASE7506 MP3 SQL Repair Agent controller (Harness V1/V2/V3-Cascade)."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

_SUBMISSION_DIR = str(Path(__file__).resolve().parent)
if _SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, _SUBMISSION_DIR)

from ast_pool import build_broad_ast_pool
from ast_repair import (
    _parse_sqlite,
    _table_sig,
    detect_spurious_wakeup_defect,
    generate_ast_repairs,
    generate_probe_repairs,
    propose_skeleton_probe,
    propose_value_probe,
    rewrite_sql_with_probes,
    score_passing_candidate,
    select_best_candidate,
)
from ranker import (
    LinearRankerBundle,
    column_match_score,
    extract_vector2_features,
    peer_relative_penalty,
    simplicity_prior,
)

_RANKER_BUNDLE = LinearRankerBundle()

MAX_SQL_BYTES = 16000
MAX_RETRIEVE_CHARS = 2000

DEFAULT_CONFIG: dict[str, Any] = {
    "version": "v3_lora",
    "enable_ast_repair": True,
    "enable_spurious_guard": True,
    "enable_llm_refiner": True,
    "use_lora": True,
}


def normalize_sql(sql: str) -> str:
    """Normalize whitespace, trailing semicolons, and case for SQL deduplication."""
    return re.sub(r"\s+", " ", sql.strip()).rstrip(";").strip().lower()


def is_valid_sql_payload(sql: Any) -> bool:
    """Verify SQL satisfies non-empty and UTF-8 byte-length tool constraints."""
    return (
        isinstance(sql, str)
        and bool(sql.strip())
        and len(sql.encode("utf-8")) <= MAX_SQL_BYTES
    )


def parse_schema(raw_schema: list[dict[str, Any]]) -> dict[str, Any]:
    """Build case-insensitive table/column lookup tables and undirected FK graph."""
    tables: dict[str, dict[str, Any]] = {}
    col_to_tables: dict[str, list[str]] = defaultdict(list)
    fk_graph: dict[str, list[dict[str, str]]] = defaultdict(list)

    for tbl in raw_schema:
        tname = tbl["name"]
        cols = {}
        pks = []
        for col in tbl.get("columns", []):
            cname = col["name"]
            cols[cname.lower()] = cname
            col_to_tables[cname.lower()].append(tname)
            if col.get("pk", 0):
                pks.append(cname)
        tables[tname.lower()] = {
            "name": tname,
            "columns": cols,
            "pks": pks,
            "foreign_keys": tbl.get("foreign_keys", []),
        }
        for fk in tbl.get("foreign_keys", []):
            ref_table = fk["table"]
            from_col = fk["from"]
            to_col = fk["to"]
            fk_graph[tname.lower()].append(
                {
                    "from_table": tname,
                    "from_col": from_col,
                    "to_table": ref_table,
                    "to_col": to_col,
                }
            )
            fk_graph[ref_table.lower()].append(
                {
                    "from_table": ref_table,
                    "from_col": to_col,
                    "to_table": tname,
                    "to_col": from_col,
                }
            )

    return {
        "raw": raw_schema,
        "tables": tables,
        "col_to_tables": dict(col_to_tables),
        "fk_graph": dict(fk_graph),
    }


def build_retrieve_query(question: str, issue_sql: str) -> str:
    """Build a bounded retrieval query mixing question keywords and SQL structure."""
    clean_q = re.sub(r"[\x00-\x1f\x7f]+", " ", question).strip()
    clean_sql = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f]+", " ", issue_sql)).strip()
    query = f"{clean_q} {clean_sql}".strip()
    return query[:MAX_RETRIEVE_CHARS] or "SELECT"


class SQLRepairAgent:
    """Budget-aware state machine executing perceive-plan-act SQL repair loop."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.tried_actions: set[str] = set()
        self.checked_norm_sqls: set[str] = set()
        self.candidates: list[dict[str, Any]] = []
        self.candidate_norms: set[str] = set()
        self.source_by_norm: dict[str, str] = {}

        self.has_checked_issue: bool = False
        self.schema_fetched: bool = False
        self.hints_fetched: bool = False
        self.retrieved: bool = False
        self.llm_rounds: int = 0
        self.llm_beam_queue: list[str] = []
        self.probe_done: bool = False
        self.tried_probe_cols: set[tuple[str, str]] = set()
        self.probe_meta_by_norm: dict[str, tuple[str, str]] = {}
        self.probe_results: list[dict[str, Any]] = []
        self.skeleton_probe_done: bool = False
        self.skeleton_meta_by_norm: dict[str, Any] = {}
        self.skeleton_zero_sigs: set[Any] = set()

        self.schema_info: dict[str, Any] | None = None
        self.retrieved_examples: list[dict[str, Any]] = []
        self.raw_hint_list: list[str] = []
        self.raw_hint_counts: Counter[str] = Counter()
        self.hint_queue: list[str] = []
        self.broad_pool_built: bool = False
        self.broad_pool_queue: list[tuple[str, str]] = []

    def _record_and_return(self, action: dict[str, Any], source_tag: str | None = None) -> dict[str, Any]:
        if action["tool"] != "submit":
            identity = json.dumps(action, sort_keys=True)
            self.tried_actions.add(identity)
            if action["tool"] in ("check", "execute"):
                norm = normalize_sql(action["sql"])
                self.checked_norm_sqls.add(norm)
                if source_tag is not None:
                    self.source_by_norm[norm] = source_tag
        return action

    def _action_used(self, action: dict[str, Any]) -> bool:
        return json.dumps(action, sort_keys=True) in self.tried_actions

    def _ingest_history(self, history: list[dict[str, Any]], issue_sql: str) -> None:
        """Synchronize local state from the latest observation history."""
        self.candidates = []
        self.candidate_norms = set()
        self.tried_actions = set()
        self.checked_norm_sqls = set()
        self.probe_results = []
        self.skeleton_zero_sigs = set()

        for event in history:
            action = {k: v for k, v in event.items() if k != "observation"}
            tool = action["tool"]
            if tool != "submit":
                self.tried_actions.add(json.dumps(action, sort_keys=True))
            obs = event.get("observation")

            if tool in ("check", "execute"):
                sql = action["sql"]
                norm = normalize_sql(sql)
                self.checked_norm_sqls.add(norm)
                if tool == "execute" and norm in self.skeleton_meta_by_norm:
                    self.skeleton_probe_done = True
                    sig = self.skeleton_meta_by_norm[norm]
                    if isinstance(obs, dict) and obs.get("ok") and obs.get("row_count", 0) == 0:
                        self.skeleton_zero_sigs.add(sig)
                    continue
                if tool == "execute" and norm in self.probe_meta_by_norm:
                    self.probe_done = True
                    tbl, col = self.probe_meta_by_norm[norm]
                    vals = [r[0] for r in obs.get("rows", []) if r] if isinstance(obs, dict) and obs.get("ok") else []
                    self.probe_results.append({"table": tbl, "col": col, "values": vals})
                    continue
                if sql == issue_sql:
                    self.has_checked_issue = True
                if norm not in self.candidate_norms:
                    self.candidate_norms.add(norm)
                    src = self.source_by_norm.get(norm, "issue" if sql == issue_sql else "hint")
                    self.candidates.append(
                        {
                            "sql": sql,
                            "norm": norm,
                            "source": src,
                            "order": len(self.candidates),
                            "result": obs,
                        }
                    )
            elif tool == "schema" and not self.schema_fetched:
                self.schema_fetched = True
                self.schema_info = parse_schema(obs)
            elif tool == "repair_hint" and not self.hints_fetched:
                self.hints_fetched = True
                self.raw_hint_list = [h for h in (obs or []) if is_valid_sql_payload(h)]
                self.raw_hint_counts = Counter(normalize_sql(h) for h in self.raw_hint_list)
                for raw_hint in self.raw_hint_list:
                    norm = normalize_sql(raw_hint)
                    if norm not in self.source_by_norm:
                        self.source_by_norm[norm] = "hint"
                    if norm not in self.checked_norm_sqls and norm not in {
                        normalize_sql(h) for h in self.hint_queue
                    }:
                        self.hint_queue.append(raw_hint)
            elif tool == "retrieve" and not self.retrieved:
                self.retrieved = True
                self.retrieved_examples = list(obs or [])

    def _select_best_sql(self, observation: dict[str, Any]) -> str:
        """Select final SQL using learned Step 4.7c choice_ranker, V2 robustness scorer, or V1 fallback."""
        passing = [
            c
            for c in self.candidates
            if isinstance(c.get("result"), dict) and c["result"].get("matches_example") is True
        ]
        if passing and self.config.get("enable_learned_ranker", False) and _RANKER_BUNDLE.loaded:
            if len(passing) == 1:
                return passing[0]["sql"]
            q = observation["task"]["question"]
            ex = observation["example"]
            raw_hint_table_sigs: Counter[tuple[str, ...]] = Counter()
            for rh in self.raw_hint_list or []:
                tsig = _table_sig(_parse_sqlite(rh))
                if tsig is not None:
                    raw_hint_table_sigs[tsig] += 1
            scored_list = []
            for c in passing:
                feats = extract_vector2_features(
                    c["sql"], observation["issue_sql"], q, observation["ordered"], ex, self.schema_info, c["source"]
                )
                s_base = (
                    score_passing_candidate(
                        c,
                        passing,
                        observation,
                        self.schema_info,
                        self.raw_hint_counts,
                        raw_hint_table_sigs,
                        self.probe_results,
                        self.skeleton_zero_sigs,
                    )
                    + 0.5 * peer_relative_penalty(c["sql"], passing, q)
                )
                s_learned = (
                    0.35 * math.tanh(_RANKER_BUNDLE.score_choice(feats) / 3.0)
                    + 0.08 * simplicity_prior(c["sql"])
                )
                scored_list.append((s_base + s_learned, -c["order"], c["sql"]))
            return max(scored_list)[2]

        if self.config.get("enable_spurious_guard", False):
            return select_best_candidate(
                self.candidates,
                observation,
                self.schema_info,
                self.raw_hint_counts,
                self.raw_hint_list,
                probe_results=self.probe_results,
                skeleton_zero_sigs=self.skeleton_zero_sigs,
            )

        example = observation["example"]
        exp_cols = len(example["columns"])
        exp_rows = example["row_count"]
        passing = [
            c
            for c in self.candidates
            if isinstance(c.get("result"), dict) and c["result"].get("matches_example") is True
        ]
        if passing:
            return passing[0]["sql"]

        valid_ok = [
            c
            for c in self.candidates
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

    def _next_ast_action(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        repairs = generate_ast_repairs(
            observation,
            self.schema_info,
            self.candidates,
            self.checked_norm_sqls,
            normalize_sql,
        )
        for rep_sql, src_tag in repairs:
            if not is_valid_sql_payload(rep_sql):
                continue
            action = {"tool": "check", "sql": rep_sql}
            if not self._action_used(action):
                return self._record_and_return(action, source_tag=src_tag)
        return None

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        issue_sql = observation["issue_sql"]
        remaining = observation["remaining"]
        self._ingest_history(observation.get("history", []), issue_sql)

        if remaining <= 0:
            best_sql = self._select_best_sql(observation)
            if not is_valid_sql_payload(best_sql):
                best_sql = issue_sql
            return {"tool": "submit", "sql": best_sql}

        # Step 1: check original draft (required before repair_hint)
        if not self.has_checked_issue:
            action = {"tool": "check", "sql": issue_sql}
            if not self._action_used(action):
                return self._record_and_return(action, source_tag="issue")

        # Step 2: fetch schema
        if not self.schema_fetched:
            action = {"tool": "schema"}
            if not self._action_used(action):
                return self._record_and_return(action)

        # Step 3: fetch repair_hint (always fetch to avoid premature stop on tiny public DBs)
        if not self.hints_fetched:
            action = {"tool": "repair_hint"}
            if not self._action_used(action):
                return self._record_and_return(action)

        # Step 4: retrieve similar training examples (deferred in 4.7b+ until LLM wakeup)
        use_broad_pool = bool(self.config.get("enable_broad_pool", False))
        if not use_broad_pool and not self.retrieved:
            query = build_retrieve_query(observation["task"]["question"], issue_sql)
            action = {"tool": "retrieve", "query": query}
            if not self._action_used(action):
                return self._record_and_return(action)

        if use_broad_pool and self.schema_fetched and self.hints_fetched and not self.broad_pool_built:
            self.broad_pool_built = True
            pool = build_broad_ast_pool(observation, self.schema_info, self.raw_hint_list, pool_cap=80)
            hint_norms = {normalize_sql(h) for h in self.hint_queue}
            prio_order = {
                "pool_tmpl": 0,
                "pool_simp": 1,
                "pool_basic": 2,
                "pool_proj": 3,
                "pool_qual": 4,
                "pool_agg": 5,
                "pool_name": 6,
                "pool_simp2": 7,
            }
            derived = [
                item
                for item in pool
                if item["source"] not in ("hint0", "issue", "hint_alt")
                and normalize_sql(item["sql"]) not in self.checked_norm_sqls
                and normalize_sql(item["sql"]) not in hint_norms
            ]
            if self.config.get("enable_learned_ranker", False) and _RANKER_BUNDLE.loaded:
                q = observation["task"]["question"]
                ex = observation["example"]
                exp_cols = ex.get("columns") or []

                def _pool_tier_bonus(src: str) -> float:
                    bonus = 4.0 if "hint" in src else 0.0
                    if "simp2" in src:
                        bonus -= 2.5
                    elif src.startswith("pool_tmpl"):
                        bonus += 3.0
                    elif src.startswith("pool_simp"):
                        bonus += 1.5
                    elif src.startswith("pool_proj"):
                        bonus += 1.0
                    return bonus

                derived.sort(
                    key=lambda it: (
                        _pool_tier_bonus(it["source"])
                        + math.tanh(
                            _RANKER_BUNDLE.score_priority(
                                extract_vector2_features(
                                    it["sql"], issue_sql, q, observation["ordered"], ex, self.schema_info, it["source"]
                                )
                            )
                            / 3.0
                        )
                        + 1.0 * column_match_score(it["sql"], exp_cols, exp_cols)
                    ),
                    reverse=True,
                )
            else:
                derived.sort(
                    key=lambda it: (
                        0 if "hint" in it["source"] else 1,
                        min((v for k, v in prio_order.items() if it["source"].startswith(k)), default=9),
                    )
                )
            self.broad_pool_queue = [(it["sql"], it["source"]) for it in derived]

        # Step 5+: check queued deduplicated repair hints that differ from issue_sql
        while self.hint_queue:
            cand_sql = self.hint_queue.pop(0)
            norm = normalize_sql(cand_sql)
            if norm in self.checked_norm_sqls:
                continue
            action = {"tool": "check", "sql": cand_sql}
            if not self._action_used(action):
                return self._record_and_return(action, source_tag="hint")

        # Step 5.5 (4.5b/4.6b): DB value probing with remaining>=3 budget protection
        # (Allow remaining==2 only when no hint_queue or AST repair remains and an ok=True empty/diverging trigger exists)
        allow_probe = remaining >= 3
        if not allow_probe and remaining == 2 and not self.hint_queue:
            has_ok_empty_or_pass = any(
                isinstance(c.get("result"), dict)
                and c["result"].get("ok") is True
                and (
                    c["result"].get("matches_example") is True
                    or c["result"].get("row_count", 0) == 0
                    or (c["result"].get("row_count") == 1 and c["result"].get("rows") == [[0]])
                )
                for c in self.candidates
            )
            if has_ok_empty_or_pass and (not self.config.get("enable_ast_repair", False) or self._next_ast_action(observation) is None):
                allow_probe = True

        if not self.probe_done and allow_probe:
            probe_spec = propose_value_probe(
                observation, self.schema_info, self.candidates, self.tried_probe_cols
            )
            if probe_spec is not None:
                probe_sql, tbl, col = probe_spec
                self.probe_done = True
                self.tried_probe_cols.add((tbl.lower(), col.lower()))
                self.probe_meta_by_norm[normalize_sql(probe_sql)] = (tbl, col)
                action = {"tool": "execute", "sql": probe_sql}
                if not self._action_used(action):
                    return self._record_and_return(action, source_tag="probe")

        if self.probe_results and remaining >= 1:
            for rep_sql, src_tag in generate_probe_repairs(
                self.candidates, self.schema_info, self.probe_results, self.checked_norm_sqls, normalize_sql
            ):
                if not is_valid_sql_payload(rep_sql):
                    continue
                action = {"tool": "check", "sql": rep_sql}
                if not self._action_used(action):
                    return self._record_and_return(action, source_tag=src_tag)

        passing = [
            c
            for c in self.candidates
            if isinstance(c.get("result"), dict) and c["result"].get("matches_example") is True
        ]
        if (
            self.config.get("enable_skeleton_probe", False)
            and not self.skeleton_probe_done
            and remaining >= 3
            and not self.hint_queue
            and len(passing) >= 2
        ):
            best_now = self._select_best_sql(observation)
            skel = propose_skeleton_probe(observation, passing, best_now)
            if skel is not None:
                skel_sql, bsig = skel
                self.skeleton_probe_done = True
                self.skeleton_meta_by_norm[normalize_sql(skel_sql)] = bsig
                action = {"tool": "execute", "sql": skel_sql}
                if not self._action_used(action):
                    return self._record_and_return(action, source_tag="skeleton_probe")

        use_llm = bool(self.config.get("enable_llm_refiner", False))
        reserve_for_llm = use_llm and self.llm_rounds == 0 and not passing and remaining <= 2

        # Step 6+: V2 deterministic SQLGlot AST repairs (unless reserving last call for LLM Round 1)
        if self.config.get("enable_ast_repair", False) and not reserve_for_llm:
            ast_action = self._next_ast_action(observation)
            if ast_action is not None:
                return ast_action

        # Step 6.5 (4.7b/c): Broad AST Pool exploration with asymmetric stopping threshold
        if use_broad_pool and self.broad_pool_queue and remaining >= 1 and not reserve_for_llm:
            ex_rows = observation["example"].get("row_count", 0)
            if ex_rows == 0:
                stop_target = 4
            elif passing and bool(
                detect_spurious_wakeup_defect(
                    self._select_best_sql(observation), observation, self.schema_info, passing
                )
            ):
                stop_target = 2
            else:
                stop_target = 1
            if len(passing) < stop_target:
                while self.broad_pool_queue:
                    psql, psrc = self.broad_pool_queue.pop(0)
                    if normalize_sql(psql) in self.checked_norm_sqls or not is_valid_sql_payload(psql):
                        continue
                    action = {"tool": "check", "sql": psql}
                    if not self._action_used(action):
                        return self._record_and_return(action, source_tag=psrc)

        # Drain any remaining queued beams before triggering another LLM round (stop once an LLM candidate passes)
        llm_passed = any(
            c["source"].startswith("llm")
            and isinstance(c.get("result"), dict)
            and c["result"].get("matches_example") is True
            for c in self.candidates
        )
        while (not passing or not llm_passed) and self.llm_beam_queue and remaining >= 1:
            cand_sql = self.llm_beam_queue.pop(0)
            if normalize_sql(cand_sql) in self.checked_norm_sqls:
                continue
            action = {"tool": "check", "sql": cand_sql}
            if not self._action_used(action):
                src_tag = "llm_lora" if self.config.get("use_lora", True) else "llm_base"
                return self._record_and_return(action, source_tag=src_tag)

        # Step 7: Beam-3 Lazy LLM Cascade + 2nd-Round Feedback Retry (max 2 rounds = 6 completions)
        if use_llm and self.llm_rounds < 2 and remaining >= 1:
            use_lora = bool(self.config.get("use_lora", True))
            src_tag = "llm_lora" if use_lora else "llm_base"

            if not passing:
                if use_broad_pool and not self.retrieved and remaining >= 2:
                    query = build_retrieve_query(observation["task"]["question"], issue_sql)
                    action = {"tool": "retrieve", "query": query}
                    if not self._action_used(action):
                        return self._record_and_return(action)
                from llm_refiner import propose_llm_beams

                self.llm_rounds += 1
                beams = propose_llm_beams(
                    observation=observation,
                    schema_info=self.schema_info,
                    retrieved_examples=self.retrieved_examples,
                    candidates=self.candidates,
                    use_lora=use_lora,
                    num_beams=3,
                    round_idx=self.llm_rounds,
                    probed_values=self.probe_results,
                )
                new_beams = []
                seen_q = {normalize_sql(x) for x in self.llm_beam_queue}
                for b in beams:
                    bsql = b["sql"]
                    if self.probe_results:
                        rew = rewrite_sql_with_probes(bsql, self.schema_info, self.probe_results)
                        if rew:
                            bsql = rew
                    nb = normalize_sql(bsql)
                    if (
                        is_valid_sql_payload(bsql)
                        and nb not in self.checked_norm_sqls
                        and nb not in seen_q
                    ):
                        seen_q.add(nb)
                        new_beams.append(bsql)
                self.llm_beam_queue = new_beams + self.llm_beam_queue

                while self.llm_beam_queue:
                    cand_sql = self.llm_beam_queue.pop(0)
                    if normalize_sql(cand_sql) in self.checked_norm_sqls:
                        continue
                    action = {"tool": "check", "sql": cand_sql}
                    if not self._action_used(action):
                        return self._record_and_return(action, source_tag=src_tag)

            elif self.llm_rounds == 0:
                best_so_far = self._select_best_sql(observation)
                defects = detect_spurious_wakeup_defect(
                    best_so_far, observation, self.schema_info, passing
                )
                if defects:
                    from llm_refiner import propose_llm_beams

                    self.llm_rounds += 1
                    beams = propose_llm_beams(
                        observation=observation,
                        schema_info=self.schema_info,
                        retrieved_examples=self.retrieved_examples,
                        candidates=self.candidates,
                        use_lora=use_lora,
                        num_beams=3,
                        round_idx=self.llm_rounds,
                        probed_values=self.probe_results,
                    )
                    seen_q = {normalize_sql(x) for x in self.llm_beam_queue}
                    for b in beams:
                        bsql = b["sql"]
                        if self.probe_results:
                            rew = rewrite_sql_with_probes(bsql, self.schema_info, self.probe_results)
                            if rew:
                                bsql = rew
                        nb = normalize_sql(bsql)
                        if (
                            not b["schema_valid"]
                            or not is_valid_sql_payload(bsql)
                            or nb in self.checked_norm_sqls
                            or nb in seen_q
                        ):
                            continue
                        b_defects = detect_spurious_wakeup_defect(
                            bsql,
                            observation,
                            self.schema_info,
                            [{"sql": bsql, "source": src_tag, "result": {}}],
                        )
                        if b_defects:
                            continue
                        seen_q.add(nb)
                        self.llm_beam_queue.append(bsql)

                    while self.llm_beam_queue:
                        cand_sql = self.llm_beam_queue.pop(0)
                        if normalize_sql(cand_sql) in self.checked_norm_sqls:
                            continue
                        action = {"tool": "check", "sql": cand_sql}
                        if not self._action_used(action):
                            return self._record_and_return(action, source_tag=src_tag)

        # Backup beam from Round 1 or paused AST repair if budget remains
        while self.llm_beam_queue and remaining >= 1:
            cand_sql = self.llm_beam_queue.pop(0)
            if normalize_sql(cand_sql) in self.checked_norm_sqls:
                continue
            action = {"tool": "check", "sql": cand_sql}
            if not self._action_used(action):
                src_tag = "llm_lora" if self.config.get("use_lora", True) else "llm_base"
                return self._record_and_return(action, source_tag=src_tag)

        if self.config.get("enable_ast_repair", False) and remaining >= 1:
            ast_action = self._next_ast_action(observation)
            if ast_action is not None:
                return ast_action

        # Final submission
        best_sql = self._select_best_sql(observation)
        if not is_valid_sql_payload(best_sql):
            best_sql = issue_sql
        return {"tool": "submit", "sql": best_sql}


def create_agent(checkpoint: str | Path | None = None) -> SQLRepairAgent:
    """Factory required by mp3_sql.predict."""
    config = dict(DEFAULT_CONFIG)
    if checkpoint is not None:
        ckpt_path = Path(checkpoint)
        if ckpt_path.is_file():
            loaded = json.loads(ckpt_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                config.update(loaded)
        else:
            raise FileNotFoundError(f"Checkpoint config not found: {ckpt_path}")
    return SQLRepairAgent(config=config)
