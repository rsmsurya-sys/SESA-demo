"""Transformer multimodal fusion for emergency triage (PyTorch).

Dimension fixes vs. the sketch that would not run:
- RoBERTa 768-d and PPG 128-d are projected to **512-d** (the transformer width).
  Concatenating 512 / 768 / 256 cannot enter ``nn.MultiheadAttention(embed_dim=512)``.
- ``TransformerEncoder`` / MHA use ``batch_first=True`` so tensors stay ``[B, T, C]``.
- Text takes ``input_ids`` + ``attention_mask``, not a lone mask tensor.
- Risk is ``sigmoid * 100`` so MSE is in 0–100 score units.

Not a diagnostic model — triage assistance research code only.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


SeverityName = ("low", "medium", "high")


class LiteImageEncoder(nn.Module):
    """Small CNN that mimics EfficientNet-B3's 1536-d pooled vector (no download)."""

    def __init__(self, out_dim: int = 1536) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Linear(128 * 4 * 4, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        return self.proj(torch.flatten(h, 1))


class LiteTextEncoder(nn.Module):
    """Token embedding + mean pool → 768-d (RoBERTa-shaped, no HuggingFace weights)."""

    def __init__(self, vocab_size: int = 50265, hidden: int = 768, max_len: int = 128) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, hidden, padding_idx=1)
        self.pos = nn.Embedding(max_len, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        pos = torch.arange(t, device=input_ids.device).unsqueeze(0).expand(b, t)
        h = self.norm(self.tok(input_ids) + self.pos(pos))
        mask = attention_mask.unsqueeze(-1).float()
        summed = (h * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


def _load_efficientnet_b3(pretrained: bool) -> nn.Module:
    from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    net = efficientnet_b3(weights=weights)
    net.classifier = nn.Identity()
    return net


def _load_roberta(name: str = "roberta-base") -> nn.Module:
    from transformers import RobertaModel

    return RobertaModel.from_pretrained(name)


class VitalEncoder1DCNN(nn.Module):
    """1D-CNN on RGB PPG windows ``[B, 3, 900]`` → 128-d."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, vitals: torch.Tensor) -> torch.Tensor:
        return self.net(vitals)


class MultimodalEmergencyTriage(nn.Module):
    """Cross-modal attention + 4-layer transformer → risk / severity / confidence."""

    def __init__(
        self,
        backbone: Literal["full", "lite"] = "lite",
        pretrained: bool = True,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
        vocab_size: int = 50265,
        freeze_backbones: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.d_model = d_model

        if backbone == "full":
            self.image_encoder = _load_efficientnet_b3(pretrained=pretrained)
            self.text_encoder = _load_roberta()
        else:
            self.image_encoder = LiteImageEncoder(out_dim=1536)
            self.text_encoder = LiteTextEncoder(vocab_size=vocab_size)

        self.image_proj = nn.Sequential(
            nn.Linear(1536, d_model),
            nn.LayerNorm(d_model),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(768, d_model),
            nn.LayerNorm(d_model),
        )
        self.vital_encoder = VitalEncoder1DCNN()
        # Sketch had Linear(128, 256) then informal padding; we project to d_model.
        self.vital_proj = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )

        self.modality_embed = nn.Parameter(torch.zeros(1, 3, d_model))
        nn.init.normal_(self.modality_embed, std=0.02)

        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        try:
            self.transformer_encoder = nn.TransformerEncoder(
                enc_layer, num_layers=num_layers, enable_nested_tensor=False
            )
        except TypeError:
            self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.risk_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.severity_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 3),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        if freeze_backbones and backbone == "full":
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            for p in self.text_encoder.parameters():
                p.requires_grad = False

    def _encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.backbone == "full":
            out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state[:, 0, :]
        return self.text_encoder(input_ids, attention_mask)

    def encode_modalities(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vitals: torch.Tensor,
    ) -> torch.Tensor:
        img = self.image_proj(self.image_encoder(image))
        txt = self.text_proj(self._encode_text(input_ids, attention_mask))
        vit = self.vital_proj(self.vital_encoder(vitals))
        tokens = torch.stack([img, txt, vit], dim=1)  # [B, 3, 512]
        return tokens + self.modality_embed

    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        vitals: torch.Tensor,
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        image: [B, 3, H, W]  (H=W=640 in the spec; 224 is fine for lite/smoke)
        input_ids, attention_mask: [B, L]
        vitals: [B, 3, 900]
        """
        tokens = self.encode_modalities(image, input_ids, attention_mask, vitals)
        attended, attn = self.cross_modal_attention(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=True
        )
        encoded = self.transformer_encoder(attended)
        fused = encoded.mean(dim=1)

        risk_logit = self.risk_head(fused)
        risk_score = torch.sigmoid(risk_logit) * 100.0
        severity_logits = self.severity_head(fused)
        confidence = torch.sigmoid(self.confidence_head(fused))

        out = {
            "risk_score": risk_score,
            "severity_logits": severity_logits,
            "confidence": confidence,
            "fused": fused,
        }
        if return_attention:
            out["cross_attn"] = attn  # [B, 3, 3] averaged heads
        return out


def combined_loss(
    outputs: dict[str, torch.Tensor],
    risk_true: torch.Tensor,
    severity_true: torch.Tensor,
    confidence_true: torch.Tensor,
    w_risk: float = 0.5,
    w_sev: float = 0.3,
    w_conf: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """0.5 MSE(risk) + 0.3 CE(severity) + 0.2 BCE(confidence)."""
    loss_risk = F.mse_loss(outputs["risk_score"].squeeze(-1), risk_true.float())
    loss_sev = F.cross_entropy(outputs["severity_logits"], severity_true.long())
    loss_conf = F.binary_cross_entropy(
        outputs["confidence"].squeeze(-1), confidence_true.float()
    )
    total = w_risk * loss_risk + w_sev * loss_sev + w_conf * loss_conf
    parts = {
        "loss": float(total.detach()),
        "loss_risk": float(loss_risk.detach()),
        "loss_severity": float(loss_sev.detach()),
        "loss_confidence": float(loss_conf.detach()),
    }
    return total, parts
