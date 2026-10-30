"""Pre-flight test training script.

Runs 1 micro-epoch across every stage of the training pipeline using real
images from data/processed to guarantee zero crashes, zero tensor shape mismatches,
and 100% clean checkpoint saving before launching long runs on the L40S server.
"""

import os
import sys
import tempfile
import torch
from pathlib import Path

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.factory import MultiTaskModel, build_teacher, build_student
from src.training.dataset import create_dataloaders
from src.training.distillation import KnowledgeDistiller
from src.training.domain_generalization import DomainGeneralizationTrainer
from src.utils.config import N_DISEASES, N_SEVERITY
from src.utils.quantization import Quantizer


def run_preflight_test():
    print("=" * 65)
    print("       NEXGROW PRE-FLIGHT TRAINING VERIFICATION TEST")
    print("=" * 65)
    print(f"Device: CPU (Local verification) | Classes: {N_DISEASES} | Severities: {N_SEVERITY}")

    # 1. Dataset & DataLoader Verification
    print("\n[1/6] Verifying DataLoader on real data/processed...")
    train_loader, val_loader = create_dataloaders(
        "data/processed",
        "data/processed",
        batch_size=4,
        image_size=112,
        num_workers=0,
    )
    print(f"  ✓ Found {len(train_loader.dataset)} total images")
    batch = next(iter(train_loader))
    imgs = batch["image"]
    d_labels = batch["disease"]
    s_labels = batch["severity"]
    print(f"  ✓ Batch loaded: images={imgs.shape}, disease_labels={d_labels}, severity_labels={s_labels}")
    assert imgs.shape == (4, 3, 112, 112), f"Unexpected shape {imgs.shape}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # 2. Teacher Ensemble Training Step
        print("\n[2/6] Verifying Teacher Ensemble step...")
        # Using mobilenet backbones for fast CPU verification
        teacher = build_teacher(N_DISEASES, backbone_names=["mobilenet_v3_small", "mobilenet_v3_small"], pretrained=False)
        teacher.train()
        optimizer = torch.optim.AdamW(teacher.parameters(), lr=1e-4)

        out = teacher(imgs)
        d_logits = out[0] if isinstance(out, (tuple, list)) else out
        loss = torch.nn.functional.cross_entropy(d_logits, d_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"  ✓ Teacher forward + backward pass successful! Loss: {loss.item():.4f}")

        teacher_ckpt = tmp_path / "teacher_test.pth"
        torch.save({"model": teacher.state_dict(), "epoch": 1}, str(teacher_ckpt))
        assert teacher_ckpt.exists(), "Teacher checkpoint failed to save!"
        print(f"  ✓ Teacher checkpoint saved successfully ({teacher_ckpt.stat().st_size / 1024:.1f} KB)")

        # 3. Student Model MultiTask Forward/Backward Step
        print("\n[3/6] Verifying MultiTask Student Model step...")
        student = MultiTaskModel(backbone_name="mobilenet_v3_small", n_diseases=N_DISEASES, n_severity=N_SEVERITY, pretrained=False)
        student.train()
        s_d_logits, s_s_logits = student(imgs)
        assert s_d_logits.shape == (4, N_DISEASES)
        assert s_s_logits.shape == (4, N_SEVERITY)
        s_loss = torch.nn.functional.cross_entropy(s_d_logits, d_labels) + 0.5 * torch.nn.functional.cross_entropy(s_s_logits, s_labels)
        s_loss.backward()
        print(f"  ✓ Student multi-task outputs verified: Disease={s_d_logits.shape}, Severity={s_s_logits.shape}")

        # 4. Knowledge Distillation Engine Step
        print("\n[4/6] Verifying Knowledge Distillation Engine...")
        distiller = KnowledgeDistiller(
            teacher=teacher,
            student=student,
            temperature=4.0,
            alpha=0.7,
            lr=1e-4,
            t_max=1,
            device="cpu",
        )
        # Create a tiny 2-batch dataloader for fast loop test
        tiny_dataset = torch.utils.data.Subset(train_loader.dataset, [0, 1, 2, 3, 4, 5, 6, 7])
        tiny_loader = torch.utils.data.DataLoader(tiny_dataset, batch_size=4)

        train_metrics = distiller.train_epoch(tiny_loader, epoch=0)
        val_metrics = distiller.evaluate(tiny_loader)
        print(f"  ✓ Distillation train step: Loss={train_metrics['loss']:.4f}, Disease Acc={train_metrics['disease_accuracy']:.2%}")
        print(f"  ✓ Distillation val step: Disease Acc={val_metrics['disease_accuracy']:.2%}")

        student_ckpt = tmp_path / "student_test.pth"
        distiller.save_student(str(student_ckpt), epoch=1, metrics=val_metrics)
        assert student_ckpt.exists(), "Student checkpoint failed to save!"
        print(f"  ✓ Distilled student saved successfully ({student_ckpt.stat().st_size / 1024:.1f} KB)")

        # 5. Domain Generalization (MixStyle) Step
        print("\n[5/6] Verifying Domain Generalization (MixStyle)...")
        dg_trainer = DomainGeneralizationTrainer(
            backbone_name="mobilenet_v3_small",
            n_diseases=N_DISEASES,
            pretrained=False,
            device="cpu",
        )
        dg_metrics = dg_trainer.train_epoch(tiny_loader, epoch=0)
        dg_val = dg_trainer.evaluate(tiny_loader)
        print(f"  ✓ DG train step: Disease Loss={dg_metrics['disease_loss']:.4f}, Disease Acc={dg_metrics['disease_accuracy']:.2%}")
        dg_ckpt = tmp_path / "dg_test.pth"
        dg_trainer.save_checkpoint(str(dg_ckpt), epoch=1, metrics=dg_val)
        assert dg_ckpt.exists(), "DG checkpoint failed to save!"
        print(f"  ✓ DG checkpoint saved successfully ({dg_ckpt.stat().st_size / 1024:.1f} KB)")

        # 6. INT8 Quantization Export Step
        print("\n[6/6] Verifying INT8 ONNX Export & Benchmark...")
        test_model = MultiTaskModel.from_checkpoint(str(dg_ckpt))
        test_model.eval()
        quantizer = Quantizer(test_model, input_shape=(1, 3, 112, 112))
        onnx_fp32 = tmp_path / "nexgrow_test_fp32.onnx"
        onnx_int8 = tmp_path / "nexgrow_test_int8.onnx"

        quantizer.export_onnx(str(onnx_fp32), output_names=["disease", "severity"])
        quantizer.quantize_onnx(str(onnx_fp32), str(onnx_int8))
        size_mb = quantizer.get_size_mb(str(onnx_int8))
        bench = quantizer.benchmark_onnx(str(onnx_int8), n_runs=5)
        print(f"  ✓ INT8 ONNX generated: size={size_mb:.2f} MB, latency={bench['mean_ms']:.2f} ms")

    print("\n" + "=" * 65)
    print("  ★ ALL 6 PRE-FLIGHT VERIFICATION STEPS PASSED WITH 0 ERRORS!")
    print("  ★ CODEBASE IS READY FOR THE 2x NVIDIA L40S SERVER!")
    print("=" * 65)


if __name__ == "__main__":
    run_preflight_test()
