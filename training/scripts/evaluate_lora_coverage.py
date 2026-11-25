"""Measure blind and repair Top-1 / Top-3 3-instance coverage on Validation (120 tasks)."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

os.environ.update(
    HF_HUB_OFFLINE='1',
    TRANSFORMERS_OFFLINE='1',
    TOKENIZERS_PARALLELISM='false',
)

ROOT = Path(__file__).resolve().parents[1]
COND = ROOT / 'delivery/goal201/conditions04'
COND_SUB = COND / 'submission'
COND_LIB = COND_SUB / 'lib'
BASE_DIR = ROOT / 'models/Qwen3-0.6B-c1899de'
sys.path[:0] = [str(ROOT), str(COND_LIB)]

from mp3_sql.common import load_bank, result_fingerprint, write_json
from mp3_sql.environment import Tools
from mp3_sql.score import load_cases
from mp3_sql.sqlite_tool import execute_sql

from ast_repair import has_suspicious_non_fk_join, validate_and_fix_schema_columns
from llm_refiner import _col_count_matches
from prompt_utils import build_repair_messages, extract_and_validate_sql

spec = importlib.util.spec_from_file_location('cond_lib_submission', COND_LIB / 'submission.py')
lib_sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib_sub)
parse_schema = lib_sub.parse_schema
build_retrieve_query = lib_sub.build_retrieve_query


def grade_3inst(row, sql, manifest, cases, case_root, exec_cache):
    metadata = manifest['databases'][row['db_id']]
    tables = [t['name'] for t in metadata['schema']]
    for case in cases['tasks'][row['id']]:
        path = case_root / case['path']
        key = (case['database_sha256'], sql, False)
        result = exec_cache.get(key)
        if result is None:
            result = execute_sql(path, sql, tables, False)
            if len(exec_cache) < 40000:
                exec_cache[key] = result
        if not (result['ok'] and result_fingerprint(result, row['ordered']) == case['expected']):
            return False
    return True


def generate_beams(model, tokenizer, messages, schema_info, example, expected_cols):
    import torch

    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(prompt_str, return_tensors='pt').to('cuda')
    prompt_tokens = int(inputs['input_ids'].shape[1])

    with torch.inference_mode():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=False,
            num_beams=3,
            num_return_sequences=3,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    sequences = gen_out.sequences
    seq_scores = getattr(gen_out, 'sequences_scores', None)
    valid_candidates = []
    seen_norm = set()

    for idx in range(sequences.shape[0]):
        gen_ids = sequences[idx, prompt_tokens:]
        raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        sql = extract_and_validate_sql(raw_text, schema_info)
        if not sql:
            continue
        schema_valid, fixed_sql = validate_and_fix_schema_columns(sql, schema_info, example)
        norm = re.sub(r'\s+', ' ', fixed_sql.strip()).rstrip(';').strip().lower()
        if norm in seen_norm:
            continue
        seen_norm.add(norm)
        col_ok = _col_count_matches(fixed_sql, expected_cols)
        non_fk = has_suspicious_non_fk_join(fixed_sql, schema_info)
        lprob = float(seq_scores[idx].item()) if seq_scores is not None else -float(idx)
        valid_candidates.append({
            'sql': fixed_sql,
            'schema_valid': bool(schema_valid),
            'col_count_match': bool(col_ok),
            'non_fk': bool(non_fk),
            'seq_score': lprob,
            'beam_rank': idx,
        })

    valid_candidates.sort(
        key=lambda c: (
            int(c['schema_valid']),
            int(c['col_count_match']),
            int(not c['non_fk']),
            c['seq_score'],
            -c['beam_rank'],
        ),
        reverse=True,
    )
    return valid_candidates, prompt_tokens


def summarize_mode(records, mode):
    subsets = {
        'all': records,
        'empty': [r for r in records if r['empty']],
        'nonempty': [r for r in records if not r['empty']],
        'issue_failed': [r for r in records if not r['issue_pub_ok']],
    }
    out = {}
    for sname, grp in subsets.items():
        m = [r[mode] for r in grp]
        out[sname] = {
            'n': len(grp),
            'top1_3inst': sum(1 for x in m if x['top1_3inst']),
            'top3_3inst': sum(1 for x in m if x['top3_3inst']),
            'top3_pub': sum(1 for x in m if x['top3_pub']),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter-dir', type=Path, required=True)
    p.add_argument('--tag', type=str, required=True)
    args = p.parse_args()

    import psutil
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(4)
    proc = psutil.Process()
    proc.cpu_affinity(proc.cpu_affinity()[::2][:4])

    adapter_dir = args.adapter_dir.resolve()
    assert adapter_dir.is_dir(), f'Missing adapter dir: {adapter_dir}'

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        BASE_DIR, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).to('cuda')
    model.eval()

    rows, manifest, folder = load_bank('validation')
    cases, case_root = load_cases('validation', folder)
    tools = Tools(folder, manifest, cache={})
    replay = json.loads((COND / 'evidence/validation-replay/validation.json').read_bytes())['predictions']
    exec_cache = {}
    records = []

    for i, (row, pred) in enumerate(zip(rows, replay, strict=True)):
        assert row['id'] == pred['id']
        raw_schema = tools.schema(row)
        schema_info = parse_schema(raw_schema)
        expected_cols = len(row['example'].get('columns', []))
        issue_obs = tools.query(row, row['issue_sql'], check=True)
        issue_pub_ok = bool(issue_obs.get('matches_example'))

        # 1. blind mode
        blind_msgs = build_repair_messages(
            row['question'], row['ordered'], raw_schema, row['example'], [], []
        )
        blind_beams, blind_ptoks = generate_beams(model, tokenizer, blind_msgs, schema_info, row['example'], expected_cols)
        for b in blind_beams:
            fb = tools.query(row, b['sql'], check=True)
            b['pub_ok'] = bool(fb.get('matches_example'))
            b['pass_3inst'] = grade_3inst(row, b['sql'], manifest, cases, case_root, exec_cache)

        # 2. repair mode (failed checks only + retrieved examples)
        retrieved = tools.retrieve(row, build_retrieve_query(row['question'], row['issue_sql']))
        failed_checks = [
            {
                'sql': e['sql'],
                'result': e['observation'],
                'source': 'issue' if e['sql'] == row['issue_sql'] else 'previous',
            }
            for e in pred['trace']
            if e['tool'] == 'check' and not e['observation']['matches_example']
        ]
        if not failed_checks and not issue_pub_ok:
            failed_checks = [{'sql': row['issue_sql'], 'result': issue_obs, 'source': 'issue'}]
        repair_msgs = build_repair_messages(
            row['question'], row['ordered'], raw_schema, row['example'], retrieved, failed_checks[-4:]
        )
        repair_beams, repair_ptoks = generate_beams(model, tokenizer, repair_msgs, schema_info, row['example'], expected_cols)
        for b in repair_beams:
            fb = tools.query(row, b['sql'], check=True)
            b['pub_ok'] = bool(fb.get('matches_example'))
            b['pass_3inst'] = grade_3inst(row, b['sql'], manifest, cases, case_root, exec_cache)

        records.append({
            'id': row['id'],
            'db_id': row['db_id'],
            'empty': row['example']['row_count'] == 0,
            'issue_pub_ok': issue_pub_ok,
            'blind': {
                'prompt_tokens': blind_ptoks,
                'top1_3inst': bool(blind_beams and blind_beams[0]['pass_3inst']),
                'top3_3inst': any(b['pass_3inst'] for b in blind_beams),
                'top3_pub': any(b['pub_ok'] for b in blind_beams),
                'beams': blind_beams,
            },
            'repair': {
                'prompt_tokens': repair_ptoks,
                'num_failed_checks': len(failed_checks[-4:]),
                'top1_3inst': bool(repair_beams and repair_beams[0]['pass_3inst']),
                'top3_3inst': any(b['pass_3inst'] for b in repair_beams),
                'top3_pub': any(b['pub_ok'] for b in repair_beams),
                'beams': repair_beams,
            },
        })
        if (i + 1) % 30 == 0:
            print(f'[{args.tag}] {i + 1}/120 elapsed={time.perf_counter() - started:.1f}s', flush=True)

    agg = {
        'tag': args.tag,
        'adapter_dir': str(adapter_dir),
        'elapsed_seconds': round(time.perf_counter() - started, 2),
        'gpu_peak_bytes': int(torch.cuda.max_memory_allocated()),
        'blind': summarize_mode(records, 'blind'),
        'repair': summarize_mode(records, 'repair'),
    }
    out_path = ROOT / 'outputs/plan4' / f'coverage4_{args.tag}.json'
    write_json(out_path, {'summary': agg, 'records': records})
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
