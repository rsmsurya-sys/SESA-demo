"""Train / validate / checkpoint MultimodalEmergencyTriage.

Example (lite backbones, fast):

    python -m ml.fusion.train --smoke

Full protocol (EfficientNet-B3 + RoBERTa, 10k samples, 100 epochs):

    python -m ml.fusion.train --backbone full --n 10000 --epochs 100 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.fusion.dataset import WORD_TO_ID, make_dataloaders
from ml.fusion.evaluate import evaluate_loader
from ml.fusion.model import MultimodalEmergencyTriage, combined_loss


def train_one_epoch(model, loader, optimizer, device) -> dict[str, float]:
    model.train()
    acc = {"loss": 0.0, "loss_risk": 0.0, "loss_severity": 0.0, "loss_confidence": 0.0}
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        vitals = batch["vitals"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(image, input_ids, attention_mask, vitals)
        loss, parts = combined_loss(
            out,
            batch["risk"].to(device),
            batch["severity"].to(device),
            batch["confidence"].to(device),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bs = image.size(0)
        n += bs
        for k, v in parts.items():
            acc[k] = acc.get(k, 0.0) + v * bs
    return {k: v / max(n, 1) for k, v in acc.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best: float, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val": best,
            "args": vars(args),
        },
        path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multimodal emergency triage fusion")
    p.add_argument("--backbone", choices=("lite", "full"), default="lite")
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--t-max", type=int, default=50)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="artifacts/fusion")
    p.add_argument("--smoke", action="store_true", help="Tiny run to verify the loop")
    p.add_argument("--pretrained", action="store_true", help="ImageNet/RoBERTa weights (full only)")
    p.add_argument("--freeze-backbones", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.n = 96
        args.epochs = 3
        args.batch_size = 8
        args.image_size = 64
        args.patience = 3
        args.t_max = 3
        args.backbone = "lite"

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = None
    use_tok = False
    if args.backbone == "full":
        from transformers import RobertaTokenizerFast

        tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        use_tok = True
        if args.image_size < 640 and not args.smoke:
            # Spec input is 640; keep a note in metrics file.
            pass

    train_loader, val_loader, test_loader = make_dataloaders(
        n=args.n,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        use_roberta_tokenizer=use_tok,
        tokenizer=tokenizer,
    )

    model = MultimodalEmergencyTriage(
        backbone=args.backbone,
        pretrained=args.pretrained,
        vocab_size=max(WORD_TO_ID.values()) + 8,
        freeze_backbones=args.freeze_backbones,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.t_max)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    stale = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics, _ = evaluate_loader(model, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}, **val_metrics}
        history.append(row)
        print(
            f"epoch {epoch:03d}  train {train_stats['loss']:.4f}  "
            f"val {val_metrics['val_loss']:.4f}  "
            f"MAE {val_metrics['risk_mae']:.2f}  "
            f"acc {val_metrics['severity_accuracy']:.3f}"
        )
        if val_metrics["val_loss"] < best - 1e-4:
            best = val_metrics["val_loss"]
            stale = 0
            save_checkpoint(outdir / "best.pt", model, optimizer, scheduler, epoch, best, args)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch} (patience={args.patience})")
                break

    save_checkpoint(outdir / "last.pt", model, optimizer, scheduler, epoch, best, args)

    ckpt_path = outdir / "best.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_metrics, _ = evaluate_loader(model, test_loader, device)
    report = {
        "best_val_loss": best,
        "test": {k: v for k, v in test_metrics.items() if k != "calibration"},
        "calibration": test_metrics["calibration"],
        "device": str(device),
        "backbone": args.backbone,
        "n": args.n,
        "disclaimer": (
            "Simulated labels only. Not a medical device. MAE/F1 here measure fit to "
            "the simulator, not clinical accuracy."
        ),
    }
    (outdir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (outdir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
