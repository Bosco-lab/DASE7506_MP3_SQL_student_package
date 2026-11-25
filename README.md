# DASE7506 Mini-Project 3: Execution-Guided Neuro-Symbolic SQLite Repair Agent

An execution-guided neuro-symbolic SQLite repair agent operating strictly within the 8-tool-call perceive–plan–act budget (`schema`, `check`, `execute`, `retrieve`, `repair_hint`, `submit`). The controller integrates deterministic SQLGlot AST candidate generation, a 128-feature linear ranker, offline `Qwen/Qwen3-0.6B` LoRA synthesis, and execution-grounded compositional, projection, and predicate refinement stages verified against three grading database instances per task.

## 1. Benchmark Results

| Split                              | Tasks | Correct (3-Instance) | Accuracy | Mean Tool Calls | Prediction SHA-256                                                 |
| ---------------------------------- | -----:| --------------------:| --------:| ---------------:| ------------------------------------------------------------------ |
| Official Test                      | 240   | 200                  | 83.33%   | 5.20 (max 8)    | `68f7fe905eadf5c68a563d698ef1d171052c8104b007731ddc8059dd3fcc34fe` |
| Official Validation (4-CPU)        | 120   | 118                  | 98.33%   | 5.20 (max 8)    | `633c122fca1b705418db003de0834eae0307ca1caecf3fe4e7d7ef8bda63b8e0` |
| Training Split (5-Fold DB-Grouped) | 480   | 467                  | 97.29%   | 4.86 (max 8)    | `f134df069e8b971317efb99dc9a7590c3b2b2e30b702684b50cefae19bfe3169` |

## 2. Repository Structure

```text
.
├── README.md                  # Documentation and offline reproduction guide
├── report.pdf                 # Technical report
├── NOTICE                     # CC BY-SA 4.0 and third-party attribution
├── requirements.txt           # Core CPU dependencies
├── requirements-model.txt     # Local Qwen3-0.6B and PEFT dependencies
├── freeze.json                # SHA-256 manifest for frozen runtime assets
├── submission/                # Agent controller, AST candidate pools, rankers, and overlays
├── adapter/                   # LoRA A/B weights (adapter_config.json, adapter_model.safetensors)
├── outputs/                   # Official test and validation predictions and score JSONs
├── runtime_audit/             # 4-CPU runtime metrics, sanitized LLM audits, compliance checks
├── training/                  # SFT, ranker, DPO, dual-mode LoRA records and training scripts
└── experiments/               # Dataset prior diagnosis and controlled ablation records
```

## 3. Model & Runtime Specification

- **Base Model**: `Qwen/Qwen3-0.6B` at immutable revision `c1899de289a04d12100db370d81485cdf75e47ca` (`596,049,920` parameters). Base weights are not redistributed and must be provisioned locally before offline evaluation.
- **LoRA Adapter**: `adapter/adapter_model.safetensors` and `adapter/adapter_config.json` (`10,092,544` LoRA A/B parameters; `606,142,464` total parameters, within the 1B limit).
- **Offline Loading & Resource Bounds**: Loaded once per process via `AutoTokenizer.from_pretrained(..., local_files_only=True, trust_remote_code=False)`, `AutoModelForCausalLM.from_pretrained(..., local_files_only=True, trust_remote_code=False, use_safetensors=True)`, and `PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)`. Inference runs with `--workers 1`, using at most 6 completions per task (`<= 2,048` new tokens).

## 4. Environment Setup

Evaluation assumes an unmodified copy of the official course starter directory (`mp3_sql/` and `data/` present alongside `submission/` and `adapter/`).

```bash
# 1. Create and activate the Conda environment (Python 3.12)
conda create -n <env_name> python=3.12 -y
conda activate <env_name>

# 2. Install pinned CPU and model dependencies
python -m pip install -r requirements.txt -r requirements-model.txt

# 3. Verify benchmark data integrity
python -m mp3_sql.check_data

# 4. Provision the declared base model revision locally before running offline
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca', local_dir='models/Qwen3-0.6B-c1899de')"
```

## 5. Offline Evaluation & Reproduction

```bash
conda activate <env_name>

ROOT_DIR="$(pwd -W 2>/dev/null || pwd)"
export PYTHONPATH="$ROOT_DIR/submission"
export PYTHONHASHSEED=2 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1
export MP3_BASE_MODEL="$ROOT_DIR/models/Qwen3-0.6B-c1899de"
export MP3_ADAPTER_DIR="$ROOT_DIR/adapter"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1

mkdir -p reproduced
export MP3_AUDIT_LOG="$ROOT_DIR/reproduced/audit.jsonl"

if command -v taskset >/dev/null 2>&1; then
  CPUS="$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[::2][:4])))')"
  RUN_PREFIX=(taskset -c "$CPUS")
else
  RUN_PREFIX=()
fi

# Official Test split prediction (--workers 1) and 3-instance scoring (--workers 4)
"${RUN_PREFIX[@]}" python -B -m mp3_sql.predict \
  --split test \
  --method submission/submission.py \
  --checkpoint submission/config.json \
  --workers 1 \
  --output reproduced/test.json

"${RUN_PREFIX[@]}" python -B -m mp3_sql.score \
  --split test \
  --predictions reproduced/test.json \
  --workers 4 \
  --output reproduced/test-score.json

# Official Validation split prediction and 3-instance scoring
"${RUN_PREFIX[@]}" python -B -m mp3_sql.predict \
  --split validation \
  --method submission/submission.py \
  --checkpoint submission/config.json \
  --workers 1 \
  --output reproduced/validation.json

"${RUN_PREFIX[@]}" python -B -m mp3_sql.score \
  --split validation \
  --predictions reproduced/validation.json \
  --workers 4 \
  --output reproduced/validation-score.json
```

## 6. Attribution & License

Spider-derived benchmark data and course framework modifications are distributed under **CC BY-SA 4.0** (see `NOTICE`). Base model weights (`Qwen/Qwen3-0.6B`) remain subject to their respective model license terms. Execution and SQL AST analysis rely on SQLite and SQLGlot.
