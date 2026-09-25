"""Evaluation metrics for risk (MAE/RMSE), severity (acc/F1/AUC), confidence (Brier)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.fusion.model import MultimodalEmergencyTriage, combined_loss


@torch.no_grad()
def collect_predictions(
    model: MultimodalEmergencyTriage,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    risks, risk_t = [], []
    sev_prob, sev_t = [], []
    conf, conf_t = [], []
    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        vitals = batch["vitals"].to(device)
        out = model(image, input_ids, attention_mask, vitals)
        risks.append(out["risk_score"].squeeze(-1).cpu().numpy())
        risk_t.append(batch["risk"].numpy())
        sev_prob.append(F.softmax(out["severity_logits"], dim=-1).cpu().numpy())
        sev_t.append(batch["severity"].numpy())
        conf.append(out["confidence"].squeeze(-1).cpu().numpy())
        conf_t.append(batch["confidence"].numpy())
    return {
        "risk_pred": np.concatenate(risks),
        "risk_true": np.concatenate(risk_t),
        "sev_prob": np.concatenate(sev_prob),
        "sev_true": np.concatenate(sev_t),
        "conf_pred": np.concatenate(conf),
        "conf_true": np.concatenate(conf_t),
    }


def _auc_ovr(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Macro one-vs-rest AUC without sklearn (rank method)."""
    aucs = []
    for k in range(proba.shape[1]):
        y = (y_true == k).astype(np.float64)
        if y.min() == y.max():
            continue
        scores = proba[:, k]
        order = np.argsort(scores)
        y_sorted = y[order]
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        ranks = np.arange(1, len(y) + 1)
        # ranks of positives after sorting ascending
        auc = (ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        aucs.append(float(auc))
    return float(np.mean(aucs)) if aucs else float("nan")


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray, n_class: int = 3) -> float:
    f1s = []
    for k in range(n_class):
        tp = np.sum((y_pred == k) & (y_true == k))
        fp = np.sum((y_pred == k) & (y_true != k))
        fn = np.sum((y_pred != k) & (y_true == k))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    return float(np.mean(f1s))


def calibration_curve(conf_true: np.ndarray, conf_pred: np.ndarray, n_bins: int = 10) -> dict:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(conf_pred, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    bin_pred, bin_true, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        bin_pred.append(float(conf_pred[m].mean()))
        bin_true.append(float(conf_true[m].mean()))
        counts.append(int(m.sum()))
    return {"bin_pred": bin_pred, "bin_true": bin_true, "counts": counts}


def compute_metrics(pred: dict[str, np.ndarray]) -> dict:
    rp, rt = pred["risk_pred"], pred["risk_true"]
    mae = float(np.mean(np.abs(rp - rt)))
    rmse = float(np.sqrt(np.mean((rp - rt) ** 2)))
    y_true = pred["sev_true"]
    y_prob = pred["sev_prob"]
    y_hat = y_prob.argmax(axis=1)
    acc = float((y_hat == y_true).mean())
    f1 = _f1_macro(y_true, y_hat)
    auc = _auc_ovr(y_true, y_prob)
    brier = float(np.mean((pred["conf_pred"] - pred["conf_true"]) ** 2))
    return {
        "risk_mae": mae,
        "risk_rmse": rmse,
        "severity_accuracy": acc,
        "severity_f1_macro": f1,
        "severity_auc_ovr": auc,
        "confidence_brier": brier,
        "calibration": calibration_curve(pred["conf_true"], pred["conf_pred"]),
    }


@torch.no_grad()
def evaluate_loader(
    model: MultimodalEmergencyTriage,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], dict]:
    model.eval()
    loss_sum = 0.0
    n = 0
    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        vitals = batch["vitals"].to(device)
        out = model(image, input_ids, attention_mask, vitals)
        loss, _ = combined_loss(
            out,
            batch["risk"].to(device),
            batch["severity"].to(device),
            batch["confidence"].to(device),
        )
        bs = image.size(0)
        loss_sum += float(loss) * bs
        n += bs
    pred = collect_predictions(model, loader, device)
    metrics = compute_metrics(pred)
    metrics["val_loss"] = loss_sum / max(n, 1)
    return metrics, pred
