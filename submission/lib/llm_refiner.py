"""Singleton offline LLM refiner (Qwen3-0.6B + optional LoRA) with audit logging."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any

import sqlglot
from sqlglot import exp

from ast_repair import has_suspicious_non_fk_join, validate_and_fix_schema_columns
from prompt_utils import build_repair_messages, extract_and_validate_sql

_SUBMISSION_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SUBMISSION_DIR.parent
_BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"

_MODEL_LOCK = threading.Lock()
_AUDIT_LOCK = threading.Lock()
_CACHED_KEY: tuple[str, str | None] | None = None
_CACHED_BUNDLE: tuple[Any, Any] | None = None


def _resolve_paths(use_lora: bool) -> tuple[str, str | None]:
    base_dir = os.environ.get("MP3_BASE_MODEL") or str(
        _REPO_ROOT / "models" / "Qwen3-0.6B-c1899de"
    )
    adapter_dir: str | None = None
    if use_lora:
        adapter_dir = os.environ.get("MP3_ADAPTER_DIR") or str(
            _REPO_ROOT / "outputs" / "lora_adapter"
        )
    return base_dir, adapter_dir


def get_or_load_model(use_lora: bool = True) -> tuple[Any, Any, str, str | None]:
    """Load and cache Qwen3-0.6B (and optional LoRA adapter) offline once per process."""
    global _CACHED_KEY, _CACHED_BUNDLE
    base_dir, adapter_dir = _resolve_paths(use_lora)
    key = (base_dir, adapter_dir)

    with _MODEL_LOCK:
        if _CACHED_KEY == key and _CACHED_BUNDLE is not None:
            tok, model = _CACHED_BUNDLE
            return tok, model, base_dir, adapter_dir

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(
            base_dir, local_files_only=True, trust_remote_code=False
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            base_dir,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        if use_lora and adapter_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)
        else:
            model = base_model
        model.eval()
        _CACHED_KEY = key
        _CACHED_BUNDLE = (tokenizer, model)
        return tokenizer, model, base_dir, adapter_dir


def _append_audit(record: dict[str, Any]) -> None:
    audit_path = Path(
        os.environ.get("MP3_AUDIT_LOG") or (_REPO_ROOT / "outputs" / "llm_audit_log.jsonl")
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    with _AUDIT_LOCK:
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def _col_count_matches(sql: str, expected_cols: int) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        if isinstance(tree, exp.Select):
            return len(tree.expressions) == expected_cols
        if isinstance(tree, exp.SetOperation):
            left = tree.left
            if isinstance(left, exp.Subquery):
                left = left.this
            if isinstance(left, exp.Select):
                return len(left.expressions) == expected_cols
    except Exception:
        pass
    return False


def propose_llm_beams(
    observation: dict[str, Any],
    schema_info: dict[str, Any] | None,
    retrieved_examples: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    use_lora: bool = True,
    num_beams: int = 3,
    max_new_tokens: int = 160,
    round_idx: int = 1,
    probed_values: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Generate up to `num_beams` SQL candidates, statically filter/rank by schema validity, and log."""
    if schema_info is None:
        return []

    import torch

    tokenizer, model, base_dir, adapter_dir = get_or_load_model(use_lora=use_lora)
    messages = build_repair_messages(
        question=observation["task"]["question"],
        ordered=observation["ordered"],
        raw_schema=schema_info["raw"],
        example=observation["example"],
        retrieved_examples=retrieved_examples,
        candidates=candidates,
        probed_values=probed_values,
    )
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt_str, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[1])

    started = time.perf_counter()
    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=min(max_new_tokens, 2048),
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.perf_counter() - started

    sequences = gen_out.sequences
    seq_scores = getattr(gen_out, "sequences_scores", None)
    expected_cols = len(observation["example"].get("columns", []))

    raw_beams: list[str] = []
    beam_gen_tokens: list[int] = []
    valid_candidates: list[dict[str, Any]] = []
    seen_norm: set[str] = set()

    for idx in range(sequences.shape[0]):
        gen_ids = sequences[idx, prompt_tokens:]
        # Count non-pad generated tokens
        non_pad = int((gen_ids != tokenizer.pad_token_id).sum().item())
        beam_gen_tokens.append(non_pad)
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        raw_beams.append(raw_text)

        sql = extract_and_validate_sql(raw_text, schema_info)
        if not sql:
            continue
        schema_valid, fixed_sql = validate_and_fix_schema_columns(
            sql, schema_info, observation["example"]
        )
        norm = re.sub(r"\s+", " ", fixed_sql.strip()).rstrip(";").strip().lower()
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        col_ok = _col_count_matches(fixed_sql, expected_cols)
        non_fk = has_suspicious_non_fk_join(fixed_sql, schema_info)
        lprob = float(seq_scores[idx].item()) if seq_scores is not None else -float(idx)
        valid_candidates.append(
            {
                "sql": fixed_sql,
                "schema_valid": bool(schema_valid),
                "col_count_match": bool(col_ok),
                "non_fk": bool(non_fk),
                "seq_score": lprob,
                "beam_rank": idx,
            }
        )

    valid_candidates.sort(
        key=lambda c: (
            int(c["schema_valid"]),
            int(c["col_count_match"]),
            int(not c["non_fk"]),
            c["seq_score"],
            -c["beam_rank"],
        ),
        reverse=True,
    )

    _append_audit(
        {
            "task_id": observation["task"]["id"],
            "mode": "lora" if use_lora else "base",
            "round": round_idx,
            "num_completions": int(sequences.shape[0]),
            "base_model": "Qwen/Qwen3-0.6B",
            "base_revision": _BASE_REVISION,
            "base_path": base_dir,
            "adapter_dir": adapter_dir,
            "prompt_tokens": prompt_tokens,
            "gen_tokens": beam_gen_tokens[0] if beam_gen_tokens else 0,
            "all_beam_gen_tokens": beam_gen_tokens,
            "elapsed_sec": round(elapsed, 3),
            "prompt": prompt_str,
            "raw_output": raw_beams[0] if raw_beams else "",
            "all_raw_beams": raw_beams,
            "validated_sql": valid_candidates[0]["sql"] if valid_candidates else None,
            "ranked_beams": valid_candidates,
        }
    )
    return valid_candidates


def propose_llm_sql(
    observation: dict[str, Any],
    schema_info: dict[str, Any] | None,
    retrieved_examples: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    use_lora: bool = True,
    max_new_tokens: int = 160,
) -> str | None:
    """Backward-compatible single-SQL wrapper around propose_llm_beams."""
    beams = propose_llm_beams(
        observation=observation,
        schema_info=schema_info,
        retrieved_examples=retrieved_examples,
        candidates=candidates,
        use_lora=use_lora,
        num_beams=3,
        max_new_tokens=max_new_tokens,
        round_idx=1,
    )
    return beams[0]["sql"] if beams else None
