"""Train compliant Qwen/Qwen3-0.6B LoRA A/B adapter on training SFT dataset."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, PeftModel, get_peft_model

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from mp3_sql.common import read_jsonl, write_json

BASE_REPO = "Qwen/Qwen3-0.6B"
BASE_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BASE_DIR = ROOT_DIR / "models" / "Qwen3-0.6B-c1899de"
DATA_PATH = ROOT_DIR / "outputs" / "lora_sft_train.jsonl"
ADAPTER_DIR = ROOT_DIR / "outputs" / "lora_adapter"
RECORD_PATH = ROOT_DIR / "outputs" / "lora_training_record.json"

MAX_SEQ_LEN = 768
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 2
LR = 3e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
SEED = 7506


class SFTDataset(Dataset):
    def __init__(self, samples: list[dict[str, list[int]]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


def collate_fn(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        n = len(item["input_ids"])
        pad_n = max_len - n
        input_ids.append(item["input_ids"] + [pad_id] * pad_n)
        attention_mask.append([1] * n + [0] * pad_n)
        labels.append(item["labels"] + [-100] * pad_n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--record-path", type=Path, default=RECORD_PATH)
    parser.add_argument("--grad-ckpt", action="store_true")
    args = parser.parse_args()

    max_seq_len = args.max_seq_len
    adapter_dir = args.adapter_dir
    record_path = args.record_path

    started = time.perf_counter()
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for Step 3 LoRA training")

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_DIR), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    raw_rows = read_jsonl(DATA_PATH)
    # Curate balanced non-redundant subset covering all unique tasks (~650-850 samples)
    post_views = {"post_v2_fail", "post_v2_fail_focused", "post_v2_suspect_spurious", "post_v2_alt"}
    curated_rows: list[dict] = []
    extra_per_task: dict[str, int] = {}
    for row in raw_rows:
        if row["view"] in post_views:
            curated_rows.append(row)
    for row in raw_rows:
        if row["view"] not in post_views:
            tid = row["id"]
            if extra_per_task.get(tid, 0) < 1:
                curated_rows.append(row)
                extra_per_task[tid] = extra_per_task.get(tid, 0) + 1

    encoded_samples: list[dict[str, list[int]]] = []
    lengths: list[int] = []
    truncated_count = 0

    for row in curated_rows:
        prompt_str = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(
            row["target_sql"] + tokenizer.eos_token, add_special_tokens=False
        )["input_ids"]
        total_len = len(prompt_ids) + len(target_ids)
        lengths.append(total_len)

        if total_len > max_seq_len:
            truncated_count += 1
            keep_prompt = max_seq_len - len(target_ids)
            if keep_prompt < 64:
                continue
            # Keep schema start and candidate feedback tail when trimming prompt
            head_keep = keep_prompt // 2
            tail_keep = keep_prompt - head_keep
            prompt_ids = prompt_ids[:head_keep] + prompt_ids[-tail_keep:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        encoded_samples.append({"input_ids": input_ids, "labels": labels})

    random.shuffle(encoded_samples)
    print(
        f"Prepared {len(encoded_samples)} samples (truncated={truncated_count}, "
        f"max_raw_len={max(lengths)}, mean_raw_len={sum(lengths)/len(lengths):.1f})"
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR),
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    )
    base_model.config.use_cache = False
    base_param_count = sum(p.numel() for p in base_model.parameters())

    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        modules_to_save=None,
    )
    model = get_peft_model(base_model, lora_cfg)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.to(device)

    # Strict compliance assertion: only LoRA A/B weights may be trainable
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    for name in trainable_names:
        if "lora_A" not in name and "lora_B" not in name:
            raise AssertionError(f"Non-LoRA trainable parameter detected: {name}")
        if any(bad in name for bad in ("embed", "lm_head", "bias", "norm")):
            raise AssertionError(f"Forbidden trainable parameter detected: {name}")

    adapter_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_param_count = base_param_count + adapter_param_count
    assert total_param_count < 1_000_000_000, "Parameter count exceeds 1B limit"
    print(
        f"Params: base={base_param_count:,}, adapter={adapter_param_count:,}, "
        f"total={total_param_count:,} ({100*adapter_param_count/base_param_count:.2f}%)"
    )

    dataset = SFTDataset(encoded_samples)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )
    steps_per_epoch = math.ceil(len(loader) / GRAD_ACCUM)
    total_opt_steps = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_opt_steps * WARMUP_RATIO))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_opt_steps)

    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    epoch_logs = []
    step_logs = []
    global_step = 0

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.perf_counter()
        ep_loss_sum = 0.0
        ep_batches = 0
        running_loss = 0.0
        running_count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            (loss / GRAD_ACCUM).backward()

            loss_val = float(loss.detach().item())
            ep_loss_sum += loss_val
            ep_batches += 1
            running_loss += loss_val
            running_count += 1

            if batch_idx % GRAD_ACCUM == 0 or batch_idx == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 10 == 0 or global_step == total_opt_steps:
                    avg_recent = running_loss / max(1, running_count)
                    lr_now = scheduler.get_last_lr()[0]
                    step_logs.append(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "loss": round(avg_recent, 4),
                            "lr": round(lr_now, 8),
                        }
                    )
                    print(
                        f"epoch={epoch}/{EPOCHS} step={global_step}/{total_opt_steps} "
                        f"loss={avg_recent:.4f} lr={lr_now:.6g}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_count = 0

        epoch_avg = ep_loss_sum / max(1, ep_batches)
        ep_secs = time.perf_counter() - ep_start
        epoch_logs.append(
            {"epoch": epoch, "mean_loss": round(epoch_avg, 4), "seconds": round(ep_secs, 2)}
        )
        print(f"Epoch {epoch} complete: mean_loss={epoch_avg:.4f} in {ep_secs:.2f}s", flush=True)

    peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved_mib = torch.cuda.max_memory_reserved(device) / (1024**2)

    # Save strictly adapter_model.safetensors and adapter_config.json
    adapter_dir.mkdir(parents=True, exist_ok=True)
    for old_file in adapter_dir.iterdir():
        if old_file.is_file():
            old_file.unlink()
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    for f in list(adapter_dir.iterdir()):
        if f.name not in ("adapter_model.safetensors", "adapter_config.json"):
            f.unlink()

    kept_files = sorted(f.name for f in adapter_dir.iterdir())
    assert kept_files == ["adapter_config.json", "adapter_model.safetensors"], kept_files
    adapter_bytes = sum(f.stat().st_size for f in adapter_dir.iterdir())

    # Verify offline reload on CPU
    del model, base_model
    torch.cuda.empty_cache()
    verify_base = AutoModelForCausalLM.from_pretrained(
        str(BASE_DIR), local_files_only=True, trust_remote_code=False, use_safetensors=True
    )
    verify_peft = PeftModel.from_pretrained(verify_base, str(adapter_dir), is_trainable=False)
    verify_peft.eval()

    record = {
        "base_model": BASE_REPO,
        "base_revision": BASE_REVISION,
        "base_param_count": base_param_count,
        "adapter_param_count": adapter_param_count,
        "total_param_count": total_param_count,
        "trainable_pct": round(100 * adapter_param_count / base_param_count, 4),
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": sorted(list(lora_cfg.target_modules)),
            "bias": "none",
            "modules_to_save": None,
        },
        "hyperparameters": {
            "max_seq_len": max_seq_len,
            "gradient_checkpointing": bool(args.grad_ckpt),
            "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "dtype": "bfloat16",
            "seed": SEED,
        },
        "dataset": {
            "path": str(DATA_PATH.relative_to(ROOT_DIR)),
            "num_samples": len(encoded_samples),
            "truncated_count": truncated_count,
            "max_raw_tokens": max(lengths),
            "mean_raw_tokens": round(sum(lengths) / len(lengths), 2),
        },
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_allocated_mib": round(peak_alloc_mib, 2),
        "peak_vram_reserved_mib": round(peak_reserved_mib, 2),
        "wall_seconds": round(time.perf_counter() - started, 2),
        "epoch_logs": epoch_logs,
        "step_logs": step_logs,
        "adapter_files": kept_files,
        "adapter_total_mib": round(adapter_bytes / (1024**2), 2),
        "reload_verified": True,
    }
    write_json(record_path, record)
    print(f"Training complete in {record['wall_seconds']}s; adapter={record['adapter_total_mib']} MiB")


if __name__ == "__main__":
    main()
