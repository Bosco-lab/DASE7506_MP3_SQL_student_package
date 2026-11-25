"""Fit a dual-mode LoRA adapter on conditions04 Train tasks without positive-answer leakage."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
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
COND_LIB = COND / 'submission/lib'
BASE = ROOT / 'models/Qwen3-0.6B-c1899de'
REV = 'c1899de289a04d12100db370d81485cdf75e47ca'
sys.path[:0] = [str(ROOT), str(COND_LIB)]

from mp3_sql.common import digest, load_bank, read_jsonl, write_json, write_jsonl
from mp3_sql.environment import Tools
from prompt_utils import build_repair_messages

spec = importlib.util.spec_from_file_location('cond_lib_submission', COND_LIB / 'submission.py')
lib_sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lib_sub)
build_retrieve_query = lib_sub.build_retrieve_query


def fit(holdout_fold: int | None, epochs: int, smoke_steps: int | None, out_dir: Path | None):
    import psutil
    import torch
    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    seed = 7506
    rng = random.Random(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    proc = psutil.Process()
    proc.cpu_affinity(proc.cpu_affinity()[::2][:4])
    started = time.perf_counter()

    train, manifest, data = load_bank('train')
    memory = data / 'train_memory.jsonl'
    folds = ROOT / 'outputs/plan2_p3/folds.json'
    replay = COND / 'evidence/train-replay/train.json'
    gold = {r['id']: r['gold_sql'] for r in read_jsonl(memory)}
    groups = json.loads(folds.read_bytes())
    pred = json.loads(replay.read_bytes())['predictions']
    assert len(train) == len(gold) == len(pred) == 480

    tools = Tools(data, manifest, cache={})

    if out_dir is not None:
        out = out_dir.resolve()
    else:
        name = 'train4_all' if holdout_fold is None else f'train4_fold{holdout_fold}'
        out = ROOT / 'outputs/plan4' / name
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    views, samples = [], []
    skipped_over_2048 = 0
    skipped_no_failed_repair = 0

    for row, p in zip(train, pred, strict=True):
        tid, db = row['id'], row['db_id']
        assert tid == p['id'] and tid.startswith('train-')
        schema = manifest['databases'][db]['schema']
        target = gold[tid].strip().rstrip(';')

        # Only keep genuinely failed checks (matches_example == False) to prevent answer leakage.
        failed_checks = [
            {
                'sql': e['sql'],
                'result': e['observation'],
                'source': 'issue' if e['sql'] == row['issue_sql'] else 'previous',
            }
            for e in p['trace']
            if e['tool'] == 'check' and not e['observation']['matches_example']
        ]
        retrieved = tools.retrieve(row, build_retrieve_query(row['question'], row['issue_sql']))

        mode_specs = [('blind', schema, [], [])]
        if failed_checks:
            mode_specs.append(('repair', rng.sample(schema, len(schema)), retrieved, failed_checks[-4:]))
        else:
            skipped_no_failed_repair += 1

        for mode, raw, ret_ex, cands in mode_specs:
            messages = build_repair_messages(row['question'], row['ordered'], raw, row['example'], ret_ex, cands)
            prompt = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            answer = tok.encode(target, add_special_tokens=False) + [tok.eos_token_id]
            if len(prompt) + len(answer) > 2048:
                skipped_over_2048 += 1
                continue
            views.append(dict(id=tid, db=db, mode=mode, messages=messages, target_sql=target))
            samples.append((prompt, answer))

    train_idx = [i for i, v in enumerate(views) if groups[v['db']] != holdout_fold]
    hold_idx = [i for i, v in enumerate(views) if groups[v['db']] == holdout_fold]
    fit_dbs = {views[i]['db'] for i in train_idx}
    hold_dbs = {views[i]['db'] for i in hold_idx}
    assert not (fit_dbs & hold_dbs)
    write_jsonl(out / 'views.jsonl', views)

    base = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16
    )
    nbase = sum(p.numel() for p in base.parameters())
    modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, target_modules=modules,
        task_type='CAUSAL_LM', bias='none', revision=REV
    )
    model = get_peft_model(base, config).to('cuda')
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

    trainable = {k: p for k, p in model.named_parameters() if p.requires_grad}
    assert all(re.search(r'\.lora_[AB]\.default\.weight$', k) for k in trainable)

    accumulate, lr = 16, 3e-4
    steps_per_epoch = math.ceil(len(train_idx) / accumulate)
    total_steps = epochs * steps_per_epoch if smoke_steps is None else smoke_steps
    warmup_steps = max(1, round(0.05 * total_steps))
    opt = torch.optim.AdamW(trainable.values(), lr=lr, weight_decay=0.01)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    record = dict(
        base='Qwen/Qwen3-0.6B',
        revision=REV,
        base_parameters=nbase,
        adapter_param_count=sum(p.numel() for p in trainable.values()),
        trainable=list(trainable),
        hyperparameters=dict(
            epochs=epochs, lr=lr, accumulation=accumulate, seed=seed, max_length=2048,
            schedule='cosine', warmup_steps=warmup_steps, total_steps=total_steps,
            gradient_checkpointing=True,
        ),
        dataset_stats=dict(
            total_views=len(views),
            skipped_over_2048=skipped_over_2048,
            skipped_no_failed_repair=skipped_no_failed_repair,
            fit_views=dict(Counter(views[i]['mode'] for i in train_idx)),
            hold_views=dict(Counter(views[i]['mode'] for i in hold_idx)),
        ),
        fit_tasks=len({views[i]['id'] for i in train_idx}),
        holdout_tasks=len({views[i]['id'] for i in hold_idx}),
        fit_databases=sorted(fit_dbs),
        holdout_databases=sorted(hold_dbs),
        versions={n: version(n) for n in ('torch', 'peft', 'transformers', 'safetensors', 'accelerate')},
        provenance={
            p.relative_to(ROOT).as_posix(): digest(p)
            for p in (
                memory, folds, replay, data / 'train.jsonl', data / 'manifest.json',
                Path(__file__), COND_LIB / 'prompt_utils.py', out / 'views.jsonl',
            )
        },
        checkpoints=[],
    )
    write_json(out / 'training.json', record)

    def loss(i):
        prompt, answer = samples[i]
        ids = torch.tensor([prompt + answer[:-1]], device='cuda')
        logits = model(input_ids=ids, logits_to_keep=len(answer), use_cache=False).logits[0]
        targets = torch.tensor(answer, device='cuda')
        return torch.nn.functional.cross_entropy(logits.float(), targets)

    best_score = float('inf')
    opt_steps = 0
    for epoch in range(1, epochs + 1):
        rng.shuffle(train_idx)
        model.train()
        total = 0.0
        seen = 0
        for start in range(0, len(train_idx), accumulate):
            batch = train_idx[start:start + accumulate]
            opt.zero_grad(set_to_none=True)
            for i in batch:
                l = loss(i)
                (l / len(batch)).backward()
                total += l.detach().item()
                seen += 1
            torch.nn.utils.clip_grad_norm_(trainable.values(), 1.0)
            opt.step()
            sched.step()
            opt_steps += 1
            if start % 128 == 0 or smoke_steps is not None:
                print(epoch, seen, len(train_idx), round(total / seen, 4), f'lr={sched.get_last_lr()[0]:.6f}', flush=True)
            if smoke_steps is not None and opt_steps >= smoke_steps:
                break

        if smoke_steps is not None:
            record['smoke_complete'] = True
            record['gpu_peak_bytes'] = int(torch.cuda.max_memory_allocated())
            write_json(out / 'training.json', record)
            print('SMOKE_OK', record['gpu_peak_bytes'], flush=True)
            return

        model.eval()
        with torch.no_grad():
            hold = sum(loss(i).item() for i in hold_idx) / len(hold_idx) if hold_idx else None
        train_nll = total / len(train_idx)
        entry = dict(
            epoch=epoch,
            training_nll=train_nll,
            holdout_nll=hold,
            wall_seconds=time.perf_counter() - started,
            gpu_peak_bytes=torch.cuda.max_memory_allocated(),
        )

        # Save every epoch so either epoch1 or epoch2 can be inspected or selected cleanly.
        save = out / f'epoch{epoch}'
        model.save_pretrained(save, safe_serialization=True, save_embedding_layers=False)
        (save / 'README.md').unlink(missing_ok=True)
        path = save / 'adapter_config.json'
        cfg = json.loads(path.read_bytes())
        cfg.update(base_model_name_or_path='Qwen/Qwen3-0.6B', revision=REV)
        write_json(path, cfg)
        tensor = save / 'adapter_model.safetensors'
        with safe_open(tensor, framework='pt') as f:
            assert all(re.search(r'\.lora_[AB]\.weight$', k) for k in f.keys())
            assert sum(f.get_tensor(k).numel() for k in f.keys()) == record['adapter_param_count']

        cur_score = hold if hold is not None else train_nll
        if cur_score < best_score:
            best_score = cur_score
            record.update(selected=save.name, adapter_sha256={p.name: digest(p) for p in (path, tensor)})
        record['checkpoints'].append(entry)
        write_json(out / 'training.json', record)
        print(entry, flush=True)

    record['complete'] = True
    write_json(out / 'training.json', record)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--holdout-fold', type=int, choices=range(5))
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--smoke-steps', type=int)
    p.add_argument('--out-dir', type=Path)
    args = p.parse_args()
    fit(args.holdout_fold, args.epochs, args.smoke_steps, args.out_dir)
