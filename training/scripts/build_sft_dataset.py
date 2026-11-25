"""Build compliant multi-view SFT dataset from ONLY the 480 training tasks.

Strictly uses only:
  - mp3_sql/data/train.jsonl
  - mp3_sql/data/train_memory.jsonl (gold_sql + retrieval memory)
  - mp3_sql/data/repair_hints.json
  - mp3_sql/data/manifest.json (schemas & training public databases only)
Never reads validation, test, or any grading references.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "submission"))

from mp3_sql.common import load_bank, read_jsonl, write_json, write_jsonl
from mp3_sql.environment import Tools, rollout
from mp3_sql.predict import load_factory
from prompt_utils import build_repair_messages


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).rstrip(";").strip().lower()


def clean_gold_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip()).rstrip(";").strip()


def main() -> None:
    started = time.perf_counter()
    data_dir = ROOT_DIR / "mp3_sql" / "data"
    rows, manifest, folder = load_bank("train", data_dir)
    train_memory = read_jsonl(folder / "train_memory.jsonl")
    gold_map = {item["id"]: clean_gold_sql(item["gold_sql"]) for item in train_memory}
    hints_doc = json.loads((folder / "repair_hints.json").read_text(encoding="utf-8"))

    tools = Tools(folder, manifest)
    ckpt_path = ROOT_DIR / "submission" / "config.json"
    factory = load_factory(ROOT_DIR / "submission" / "submission.py")

    def run_v2_train(row: dict) -> dict:
        agent = factory(ckpt_path)
        return rollout(row, tools, agent)

    with ThreadPoolExecutor(max_workers=4) as pool:
        rollouts = list(pool.map(run_v2_train, rows))

    sft_rows: list[dict] = []
    view_counts: Counter[str] = Counter()
    v2_pub_pass_count = 0
    v2_pub_fail_count = 0
    suspect_tiny_pub_count = 0

    for row, pred in zip(rows, rollouts):
        tid = row["id"]
        gold_sql = gold_map[tid]
        norm_gold = normalize_sql(gold_sql)
        raw_schema = tools.schema(row)
        raw_hints_norm = {
            normalize_sql(h) for h in hints_doc[tid]["suggestions"] if isinstance(h, str)
        }

        retrieved = []
        checks: list[dict] = []
        for idx, ev in enumerate(pred["trace"]):
            if ev["tool"] == "retrieve":
                retrieved = ev["observation"]
            elif ev["tool"] == "check":
                sql = ev["sql"]
                if idx == 0:
                    src = "issue"
                elif normalize_sql(sql) in raw_hints_norm:
                    src = "hint"
                else:
                    src = "ast"
                checks.append({"sql": sql, "source": src, "result": ev["observation"]})

        submitted_norm = normalize_sql(pred["sql"])
        v2_passed_pub = any(
            c["result"].get("matches_example") is True for c in checks
        )
        if v2_passed_pub:
            v2_pub_pass_count += 1
        else:
            v2_pub_fail_count += 1

        pub_rc = row["example"]["row_count"]
        is_suspect_spurious = (
            v2_passed_pub and pub_rc <= 1 and submitted_norm != norm_gold
        )
        if is_suspect_spurious:
            suspect_tiny_pub_count += 1

        def add_sample(view_name: str, cand_list: list[dict]) -> None:
            msgs = build_repair_messages(
                question=row["question"],
                ordered=row["ordered"],
                raw_schema=raw_schema,
                example=row["example"],
                retrieved_examples=retrieved,
                candidates=cand_list,
            )
            sft_rows.append(
                {
                    "id": tid,
                    "db_id": row["db_id"],
                    "view": view_name,
                    "messages": msgs,
                    "target_sql": gold_sql,
                }
            )
            view_counts[view_name] += 1

        # View A: Post-V2 trajectory -> gold_sql
        if not v2_passed_pub:
            # Oversample hard V2 public-failing tasks 2x (full trace + failed-only view)
            add_sample("post_v2_fail", checks)
            failed_only = [c for c in checks if not c["result"].get("matches_example")]
            if failed_only:
                add_sample("post_v2_fail_focused", failed_only[-2:])
        elif is_suspect_spurious:
            add_sample("post_v2_suspect_spurious", checks)
        elif submitted_norm != norm_gold:
            add_sample("post_v2_alt", checks)

        # View B: issue_sql-only feedback -> gold_sql (when issue failed or differs from gold)
        if checks:
            issue_c = checks[0]
            if (not issue_c["result"].get("matches_example")) or (
                normalize_sql(issue_c["sql"]) != norm_gold
            ):
                add_sample("issue_only", [issue_c])

        # View C: issue_sql + each unique repair_hint failing or differing from gold
        for c in checks[1:]:
            if c["source"] == "hint" and (
                (not c["result"].get("matches_example"))
                or (normalize_sql(c["sql"]) != norm_gold and pub_rc <= 1)
            ):
                add_sample("hint_refine", [checks[0], c])

    out_jsonl = ROOT_DIR / "outputs" / "lora_sft_train.jsonl"
    write_jsonl(out_jsonl, sft_rows)

    stats = {
        "split": "train_only",
        "source_tasks": len(rows),
        "v2_train_public_pass": v2_pub_pass_count,
        "v2_train_public_fail": v2_pub_fail_count,
        "v2_train_suspect_spurious_diff_from_gold": suspect_tiny_pub_count,
        "total_sft_examples": len(sft_rows),
        "view_counts": dict(view_counts),
        "unique_tasks_in_sft": len({r["id"] for r in sft_rows}),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "provenance": [
            "mp3_sql/data/train.jsonl",
            "mp3_sql/data/train_memory.jsonl",
            "mp3_sql/data/repair_hints.json",
            "mp3_sql/data/manifest.json (train schemas + train public sqlite DBs only)",
        ],
    }
    stats_path = ROOT_DIR / "outputs" / "lora_data_stats.json"
    write_json(stats_path, stats)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
