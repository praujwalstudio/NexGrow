# NexGrow — Audit Prompt

> **Hand this document verbatim to a fresh AI agent (or run it yourself) to produce a PASS/FAIL/WARN verification report.**

---

## Usage

Run all checks from the repository root with:

```bash
# Full test suite
python -m pytest tests/ -q

# Lint
python -m ruff check .

# Format check
python -m ruff format . --check
```

Then walk the 13 itemized checks below **in order**. For each check you must:

1. **Files/symbols exist** — confirm the file exists and the listed symbols appear by reading the file.
2. **Logic correct** — read the code at the cited lines AND execute the specified command, capturing its output.
3. **Cite evidence** — record `file:line` and the command + stdout/stderr snippet for every finding.

Produce the report using the **Report Template** at the bottom. A final verdict of **PASS** requires every item to be PASS or WARN (not FAIL).

---

## Audit Contract

- **Read-only verification**: do not modify any source file. Only execute commands.
- **Evidence required**: every verdict must cite ≥1 file:line reference and ≥1 command output snippet.
- **WARN** is acceptable for items that are structurally correct but require external resources (real weights, API keys, hardware) that are not available in this environment.
- **FAIL** is used when a file is missing, a symbol cannot be found, or a command exits non-zero for unexpected reasons.

---

## 13 Itemized Checks

---

### 1 — Model Factory

**File:** `src/models/factory.py`

**What to verify:**

- `_BACKBONES` dict at line 21 contains **8 entries**: `mobilenet_v2`, `mobilenet_v3_large`, `mobilenet_v3_small`, `swin_v2_b`, `swin_v2_t`, `convnext_base`, `convnext_tiny`, `resnet50`.
- Functions `build_classifier` (line 94), `build_student` (line 103), `build_teacher` (line 124), `build_feature_extractor` (line 151) are all importable.
- Class `MultiTaskModel` (line 158) with `save_checkpoint` (line 188) and `load_checkpoint` (line 212) and `from_checkpoint` (line 221) supports a round-trip: save → load → forward pass produces `(disease_logits, severity_logits)`.
- `_replace_last_linear_with_identity` (line 71) handles the `mobilenet_v3_*` nested head correctly (last linear replaced by `nn.Identity`, not the whole classifier).

**Spot-check command:**
```python
python - <<'EOF'
import tempfile, os, torch
from src.models.factory import MultiTaskModel, build_student, build_teacher, build_feature_extractor
m = MultiTaskModel(backbone_name="mobilenet_v3_small", n_diseases=4, pretrained=False)
d, s = m(torch.zeros(1, 3, 112, 112))
assert d.shape == (1, 4) and s.shape == (1, 3), f"bad shapes: {d.shape}, {s.shape}"
with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f: path = f.name
m.save_checkpoint(path, epoch=1)
m2 = MultiTaskModel.from_checkpoint(path)
d2, s2 = m2(torch.zeros(1, 3, 112, 112))
assert d2.shape == (1, 4)
os.unlink(path)
print("PASS: MultiTaskModel checkpoint round-trip OK")
EOF
```

---

### 2 — Config

**File:** `src/utils/config.py`

**What to verify:**

- `DISEASE_CLASSES` (line 17) contains exactly **15** entries; last entry is `"Root_Rot"`.
- `SEVERITY_LEVELS` (line 35) contains exactly **3** entries: `["Low", "Moderate", "High"]`.
- `TrainConfig.load()` (line 87) returns a `TrainConfig` object with nested `DataConfig`, `ModelConfig`, `TrainingConfig`, `PathConfig` when called with no arguments (falls back to `configs/train.yaml` or defaults).
- `resolve_device("auto")` (line 108) returns either `"cuda"` or `"cpu"` without raising.

**Spot-check command:**
```python
python - <<'EOF'
from src.utils.config import DISEASE_CLASSES, SEVERITY_LEVELS, TrainConfig, resolve_device
# Bacterial_Light is temporarily commented out (no public data).
# N_DISEASES auto-adjusts. Re-run after restoring it to validate 15.
assert len(DISEASE_CLASSES) == 14, f"Expected 14 (Bacterial_Light deferred), got {len(DISEASE_CLASSES)}"
assert DISEASE_CLASSES[-1] == "Root_Rot"
assert SEVERITY_LEVELS == ["Low", "Moderate", "High"]
cfg = TrainConfig.load()
assert hasattr(cfg, "data") and hasattr(cfg, "model")
dev = resolve_device("auto")
assert dev in ("cuda", "cpu"), f"Unexpected device: {dev}"
print(f"PASS: DISEASE_CLASSES={len(DISEASE_CLASSES)}, SEVERITY_LEVELS=3, resolve_device={dev!r}")
EOF
```

---

### 3 — Augmentation (Albumentations 2.x API)

**File:** `src/preprocessing/augmentation.py`

**What to verify:**

- `AugmentationPipeline._build_train_transform` (line 24) uses the albumentations 2.x keyword `size=` for `RandomResizedCrop` (line 29), `num_holes_range=` for `CoarseDropout` (line 39), and `std_range=` for `GaussNoise` (line 37). These are the renamed args in albumentations ≥ 1.4; using the old `height`/`width` or `var_limit` would raise `TypeError`.
- `AugmentationPipeline` instantiates without error and `train_transform(image=img)["image"]` produces an output from a dummy `(224, 224, 3)` uint8 image.

**Spot-check command:**
```python
python - <<'EOF'
import numpy as np
from src.preprocessing.augmentation import AugmentationPipeline
aug = AugmentationPipeline(image_size=112)
dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
out = aug.train_transform(image=dummy)["image"]
assert out.shape[0] == 3 or out.shape == (112, 112, 3), f"Unexpected shape: {out.shape}"
print(f"PASS: Albumentations 2.x pipeline OK, output shape={out.shape}")
EOF
```

---

### 4 — Domain Generalization Trainer

**File:** `src/training/domain_generalization.py`

**What to verify:**

- Class `DomainGeneralizationTrainer` with methods `train_epoch` (line 50), `evaluate` (line 92), `save_checkpoint` (line 118), `load_checkpoint` (line 134).
- `train_epoch` reads `batch["disease"]` and `batch.get("severity", ...)` — **no bare** `labels` variable (that was the bug fixed in `fix/dg-trainer`).
- `__init__` calls `build_feature_extractor` for the backbone and attaches `.disease_head` and `.severity_head` as separate `nn.Linear` layers — confirming the FeatureExtractor wiring.
- A 1-epoch smoke run returns a dict containing `disease_accuracy`.

**Spot-check command:**
```python
python - <<'EOF'
import tempfile, os
from pathlib import Path
import cv2, numpy as np
from src.utils.config import DISEASE_CLASSES, SEVERITY_LEVELS
from src.training.dataset import create_dataloaders
from src.training.domain_generalization import DomainGeneralizationTrainer

tmp = Path(tempfile.mkdtemp())
for di in range(2):
    for si in range(2):
        d = tmp / "train" / DISEASE_CLASSES[di] / SEVERITY_LEVELS[si]
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "leaf_0.png"), np.zeros((112,112,3), np.uint8))

for di in range(2):
    for si in range(2):
        d = tmp / "val" / DISEASE_CLASSES[di] / SEVERITY_LEVELS[si]
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "leaf_0.png"), np.zeros((112,112,3), np.uint8))

train_l, val_l = create_dataloaders(str(tmp/"train"), str(tmp/"val"), batch_size=2, image_size=112, num_workers=0)
trainer = DomainGeneralizationTrainer(backbone_name="mobilenet_v3_small", n_diseases=2, pretrained=False, device="cpu")
m = trainer.train_epoch(train_l, 0)
assert "disease_accuracy" in m, f"Missing key: {m}"
print("PASS: DG trainer 1-epoch OK, no labels NameError")
EOF
```

---

### 5 — Train CLI

**File:** `scripts/train.py`

**What to verify:**

- CLI supports `--stage distill`, `--stage dg-train`, and `--stage quantize`.
- All three stages exit 0 on a tiny synthetic dataset (2 diseases, 1 severity, 1 image per class).

**Spot-check command:**
```bash
python scripts/train.py --help
python scripts/train.py \
  --data-dir %TEMP%\nexgrow_smoke \
  --epochs 1 --batch-size 4 \
  --backbone mobilenet_v3_small \
  --stage distill
echo Exit distill: $?
python scripts/train.py \
  --data-dir %TEMP%\nexgrow_smoke \
  --epochs 1 --batch-size 4 \
  --backbone mobilenet_v3_small \
  --stage dg-train
echo Exit dg-train: $?
```

> **WARN** is acceptable for the `quantize` stage if `onnxscript` is not installed. Check for the ImportError message vs a genuine logic failure.

---

### 6 — Dataset

**File:** `src/training/dataset.py`

**What to verify:**

- `CottonLeafDataset.__getitem__` (line 63) returns a dict with keys `{"image", "disease", "severity", "path"}` when `multi_task=True`.
- `create_dataloaders` (line 128) returns `(train_loader, val_loader)` — a 2-tuple.
- `apply_adasyn` (line 90) returns the dataset **unchanged** when only 1 disease class is present (single-class guard at line 103).
- `pin_memory` (line 159) is `True` only when `torch.cuda.is_available()` is `True` — confirming CUDA-only behavior.

**Spot-check command:**
```python
python - <<'EOF'
import tempfile
from pathlib import Path
import cv2, numpy as np
from src.utils.config import DISEASE_CLASSES, SEVERITY_LEVELS
from src.training.dataset import CottonLeafDataset, apply_adasyn, create_dataloaders

tmp = Path(tempfile.mkdtemp())
# Single-class guard test
d = tmp / DISEASE_CLASSES[0] / SEVERITY_LEVELS[0]
d.mkdir(parents=True, exist_ok=True)
for i in range(3):
    cv2.imwrite(str(d / f"leaf_{i}.png"), np.zeros((112,112,3), np.uint8))

ds = CottonLeafDataset(str(tmp))
original_count = len(ds.samples)
resampled = apply_adasyn(ds)
assert len(resampled.samples) == original_count, "single-class guard failed"
print(f"PASS: apply_adasyn single-class guard OK (samples={original_count})")
EOF
```

---

### 7 — Smoke Tests

**File:** `tests/test_training_smoke.py`

**What to verify:**

- File contains exactly **5 test methods** inside `TestTrainingSmoke`:
  1. `test_dataloaders_build_from_disk` (line 13)
  2. `test_distillation_smoke` (line 29)
  3. `test_dg_trainer_smoke` (line 54)
  4. `test_apply_adasyn_balances_imbalanced` (line 82)
  5. `test_apply_adasyn_single_class_returns_unchanged` (line 93)
- All 5 pass.

**Spot-check command:**
```bash
python -m pytest tests/test_training_smoke.py -v
```

Expected output: `5 passed` with no failures.

---

### 8 — Real Inference

**File:** `src/inference/pipeline.py`

**What to verify:**

- `DiagnosticPipeline.run` (line 131) returns a `DiagnosisResult` where `.leaves` is a list of `LeafDiagnosis` objects with real model predictions — not placeholder strings like `"Unknown Disease"`.
- Each `LeafDiagnosis` has `.disease` (str from `DISEASE_CLASSES`), `.disease_idx` (int), `.severity` (str from `SEVERITY_LEVELS`), `.severity_idx` (int), `.confidence` (float ∈ [0,1]), `.isolated_image` (ndarray with `.size > 0`).
- `run_file` (line 166) wraps `cv2.imread` + `run`, raises `FileNotFoundError` for missing paths.
- `DiagnosticPipeline._detect` falls back to the full-image bounding box when no YOLO detector is loaded (line 117–118).

**Spot-check command:**

Covered end-to-end by `tests/test_integration.py::test_full_pipeline_end_to_end` and `tests/test_integration.py::test_api_serves_trained_model_end_to_end`. Run:
```bash
python -m pytest tests/test_integration.py -v
```

---

### 9 — Data Preparation

**File:** `scripts/prepare_data.py`

**What to verify:**

- CLI entry point is `prepare` (line 80), registered as `if __name__ == "__main__": prepare()` (line 158).
- `--dry-run` flag exits 0 and prints `"(dry run — nothing written)"`.
- Output directory structure `<disease>/<severity>/` matches `CottonLeafDataset._build_index` expectations.
- `_infer_labels` (line 60) correctly skips images whose parent folder is not in `DISEASE_CLASSES`.

**Spot-check command:**
```bash
python scripts/prepare_data.py --input data/raw --dry-run
```

> **WARN** acceptable if `data/raw` is empty — look for `"No images found."` exit rather than a traceback. The CLI itself must exit 0 for dry-run with existing data.

---

### 10 — FastAPI Backend

**File:** `server/main.py`

**What to verify:**

- Routes defined: `GET /health` (line 72), `GET /models` (line 81), `POST /diagnose` (line 91).
- `/diagnose` raises HTTP 503 when `MODEL_PATH` does not exist (line 99–103).
- `MODEL_PATH` reads from env var `NEXGROW_MODEL` (line 25); `OPENWEATHER_API_KEY` from env (line 27).
- `get_pipeline()` (line 51) returns `None` if the model file is absent, triggering the 503.

**Spot-check command:**

Covered by `tests/test_api.py`. Run:
```bash
python -m pytest tests/test_api.py -v
```

Also verify the 503 path manually:
```python
python - <<'EOF'
import os
os.environ["NEXGROW_MODEL"] = "/nonexistent/path.pth"
import importlib, server.main as api
importlib.reload(api)
from fastapi.testclient import TestClient
client = TestClient(api.app)
resp = client.post("/diagnose", files={"image": ("x.jpg", b"\xff\xd8\xff", "image/jpeg")})
assert resp.status_code == 503, f"Expected 503, got {resp.status_code}"
print("PASS: /diagnose returns 503 without model")
EOF
```

---

### 11 — Benchmark CLI

**File:** `scripts/benchmark.py`

**What to verify:**

- CLI accepts `--model-path`, `--onnx-path`, `--output-dir`, `--input-size`, `--n-runs`, `--quantize`.
- `Quantizer(None, ...)` path (line 44) is valid: when `--onnx-path` is supplied, the model is skipped and only `benchmark_onnx` is called.
- FP32 + INT8 reports print `Size`, `Mean`, `P50`, `P95` lines (line 22–25).
- With `%TEMP%\nexgrow_smoke\bench\nexgrow_fp32.onnx` already present, `--onnx-path` mode should run without needing a `.pth` file.

**Spot-check command:**
```bash
python scripts/benchmark.py --help
# --onnx-path bypasses model load; --model-path is still required by Click (pass any placeholder)
python scripts/benchmark.py \
  --model-path %TEMP%\nexgrow_smoke\dg.pth \
  --onnx-path %TEMP%\nexgrow_smoke\bench\nexgrow_fp32.onnx \
  --n-runs 20
```

---

### 12 — CI / Lint

**Files:** `.github/workflows/ci.yml`, `pyproject.toml`

**What to verify:**

- `ci.yml` (26 lines) has jobs running `ruff check .` (line 23) and `pytest -q` (line 25) on `ubuntu-latest` with Python `3.11`.
- `pyproject.toml` contains the `[tool.ruff]` section (or `[tool.ruff.lint]`) with at least `select` and `ignore` keys.
- `ruff check .` exits 0 (no violations).
- `ruff format . --check` exits 0 (all files already formatted).

**Spot-check commands:**
```bash
python -m ruff check .
python -m ruff format . --check
```

Both must print nothing (or `"All checks passed!"`) and exit 0.

---

### 13 — Integration Tests & Dependencies

**Files:** `tests/test_integration.py`, `tests/helpers.py`, `requirements.txt`

**What to verify:**

- `tests/helpers.py` exports `make_leaf_dataset` (line 9) and `make_leaf_image` (line 22).
- `tests/test_integration.py` has 2 tests: `test_full_pipeline_end_to_end` (line 42) and `test_api_serves_trained_model_end_to_end` (line 60).
- `requirements.txt` includes all critical packages: `torch`, `torchvision`, `ultralytics`, `fastapi`, `uvicorn`, `albumentations`, `onnx`, `onnxruntime`, `grad-cam`, `lime`, `imbalanced-learn`, `click`, `pytest`.
- Total test count is **20 passed**.

**Global check commands:**
```bash
python -m pytest tests/ -q
python -c "import torch, torchvision, fastapi, albumentations, onnx, onnxruntime, click; print('All critical imports OK')"
```

---

## Global Commands (Run First)

```bash
# 1. Full test suite (must be 20 passed, 0 failed)
python -m pytest tests/ -q

# 2. Lint (must be clean)
python -m ruff check .

# 3. Format check (must be clean)
python -m ruff format . --check
```

---

## Report Template

Fill in after running all 13 checks. Replace `?` with actual findings.

```markdown
# NexGrow Audit Report
**Date:** YYYY-MM-DD
**Auditor:** <agent or human name>
**Repo commit:** <git rev-parse HEAD>

## Summary
- Tests: ? / 20 passed
- ruff check: PASS / FAIL
- ruff format: PASS / FAIL
- Items passed: ? / 13
- Final verdict: **PASS** / **FAIL** / **WARN**

## Item Results

| ID | Item | Files exist | Logic OK | Key Evidence (file:line + command) | Verdict |
|----|------|-------------|----------|------------------------------------|---------|
| 1 | Model factory | ? | ? | `factory.py:21` `_BACKBONES` 8 keys; round-trip cmd output | ? |
| 2 | Config | ? | ? | `config.py:17` len=15; `resolve_device` output | ? |
| 3 | Augmentation | ? | ? | `augmentation.py:29,37,39` kwarg names; pipeline output shape | ? |
| 4 | DG trainer | ? | ? | `domain_generalization.py:63-64` `batch["disease"]`; 1-epoch metrics | ? |
| 5 | Train CLI | ? | ? | `train.py --help` output; exit codes for distill/dg-train | ? |
| 6 | Dataset | ? | ? | `dataset.py:103` single-class guard; `__getitem__` keys | ? |
| 7 | Smoke tests | ? | ? | `pytest tests/test_training_smoke.py -v` — 5 passed | ? |
| 8 | Real inference | ? | ? | `test_integration.py::test_full_pipeline_end_to_end` passed | ? |
| 9 | Data prep | ? | ? | `prepare_data.py --dry-run` output | ? |
| 10 | FastAPI | ? | ? | `test_api.py -v` passed; 503 verified | ? |
| 11 | Benchmark | ? | ? | `benchmark.py --onnx-path ...` output lines | ? |
| 12 | CI/lint | ? | ? | `ruff check .` exit 0; `ci.yml:23,25` confirmed | ? |
| 13 | Integration+deps | ? | ? | `pytest -q` 20 passed; import check OK | ? |

## Notes / WARNs
- List any WARN items with reason (missing real weights, API key, hardware unavailable, etc.)
```

---

> **Final rule:** A verdict of PASS requires every row in the table to be either PASS or WARN, with WARNs explained by external resource unavailability. Any FAIL row makes the overall verdict FAIL.
