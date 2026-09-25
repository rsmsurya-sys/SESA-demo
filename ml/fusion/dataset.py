"""On-the-fly simulated emergency triage dataset (10k by default).

Images are generated per index (not stored) so 10,000 × 640×640 RGB is not held
in RAM. Labels are a noisy function of vitals + symptom severity so a small
model can actually learn in a smoke test.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

SYMPTOM_BANK = {
    0: [
        "mild swelling after twist, can walk with limp",
        "bruise on shin, pain is bearable",
        "sore ankle after match, no numbness",
    ],
    1: [
        "cannot bear weight, throbbing pain and swelling",
        "deep bruise and dizziness when standing",
        "open scrape with ongoing pain, feeling faint",
    ],
    2: [
        "heavy bleeding from open wound, pale and sweaty",
        "unsteady, short of breath, chest tightness after collision",
        "possible head injury, confused, rapid pulse",
    ],
}

# Tiny word-piece-ish vocab for the lite encoder (RoBERTa ids used only in `full`).
WORD_TO_ID = {"<pad>": 1, "<unk>": 3, "<s>": 0, "</s>": 2}
_next = 4
for _lines in SYMPTOM_BANK.values():
    for line in _lines:
        for w in line.replace(",", "").split():
            if w not in WORD_TO_ID:
                WORD_TO_ID[w] = _next
                _next += 1


def tokenize_lite(text: str, max_len: int = 64) -> tuple[list[int], list[int]]:
    ids = [WORD_TO_ID["<s>"]]
    for w in text.replace(",", "").split():
        ids.append(WORD_TO_ID.get(w, WORD_TO_ID["<unk>"]))
    ids.append(WORD_TO_ID["</s>"])
    ids = ids[:max_len]
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(WORD_TO_ID["<pad>"])
        mask.append(0)
    return ids, mask


def _make_image(severity: int, rng: np.random.Generator, size: int) -> np.ndarray:
    """Solid-ish injury cue: redder / larger blob ⇒ higher severity (learnable shortcut)."""
    img = rng.uniform(0.35, 0.55, size=(3, size, size)).astype(np.float32)
    cy, cx = size // 2, size // 2
    rad = int(size * (0.12 + 0.08 * severity))
    yy, xx = np.ogrid[:size, :size]
    disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= rad**2
    color = np.array([0.45 + 0.2 * severity, 0.15, 0.12], dtype=np.float32)
    img[:, disc] = color[:, None]
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def _make_ppg(hr_bpm: float, spo2: float, rr: float, rng: np.random.Generator, n: int = 900, fs: float = 30.0) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / fs
    pulse = np.sin(2 * np.pi * (hr_bpm / 60.0) * t) + 0.2 * np.sin(4 * np.pi * (hr_bpm / 60.0) * t)
    resp = 0.15 * np.sin(2 * np.pi * (rr / 60.0) * t)
    ac_scale = 0.04 + 0.002 * (spo2 - 90.0)
    r = 0.55 + ac_scale * 0.7 * pulse + resp
    g = 0.50 + ac_scale * pulse + resp
    b = 0.35 + ac_scale * 0.4 * pulse
    rgb = np.stack([r, g, b], axis=0).astype(np.float32)
    rgb += rng.normal(0, 0.01, rgb.shape).astype(np.float32)
    return rgb


class SimulatedTriageDataset(Dataset):
    def __init__(
        self,
        n: int = 10_000,
        image_size: int = 224,
        seed: int = 42,
        augment: bool = False,
        max_len: int = 64,
        use_roberta_tokenizer: bool = False,
        tokenizer=None,
    ) -> None:
        self.n = int(n)
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.augment = augment
        self.max_len = max_len
        self.use_roberta_tokenizer = use_roberta_tokenizer
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return self.n

    def _rng(self, idx: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + idx * 9973)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = self._rng(idx)
        # Class-balanced-ish severity from a 3-way draw, then vitals around it.
        severity = int(rng.integers(0, 3))
        if severity == 0:
            hr = float(rng.uniform(58, 95))
            spo2 = float(rng.uniform(96, 100))
            rr = float(rng.uniform(12, 18))
            risk = float(np.clip(rng.normal(22, 8), 0, 45))
            conf = float(np.clip(rng.uniform(0.65, 0.95), 0, 1))
        elif severity == 1:
            hr = float(rng.uniform(95, 125))
            spo2 = float(rng.uniform(92, 97))
            rr = float(rng.uniform(18, 24))
            risk = float(np.clip(rng.normal(55, 10), 35, 75))
            conf = float(np.clip(rng.uniform(0.50, 0.85), 0, 1))
        else:
            hr = float(rng.uniform(120, 170))
            spo2 = float(rng.uniform(88, 94))
            rr = float(rng.uniform(22, 30))
            risk = float(np.clip(rng.normal(82, 8), 65, 100))
            conf = float(np.clip(rng.uniform(0.40, 0.75), 0, 1))

        text = SYMPTOM_BANK[severity][int(rng.integers(0, len(SYMPTOM_BANK[severity])))]
        img = _make_image(severity, rng, self.image_size)
        if self.augment:
            if rng.random() < 0.5:
                img = img[:, :, ::-1].copy()
            img = np.clip(img * float(rng.uniform(0.85, 1.15)), 0, 1).astype(np.float32)

        vitals = _make_ppg(hr, spo2, rr, rng)

        if self.use_roberta_tokenizer and self.tokenizer is not None:
            enc = self.tokenizer(
                text,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
        else:
            ids, mask = tokenize_lite(text, self.max_len)
            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.tensor(mask, dtype=torch.long)

        return {
            "image": torch.from_numpy(img.copy()),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vitals": torch.from_numpy(vitals),
            "risk": torch.tensor(risk, dtype=torch.float32),
            "severity": torch.tensor(severity, dtype=torch.long),
            "confidence": torch.tensor(conf, dtype=torch.float32),
        }


def make_dataloaders(
    n: int = 10_000,
    batch_size: int = 32,
    image_size: int = 224,
    seed: int = 42,
    num_workers: int = 0,
    use_roberta_tokenizer: bool = False,
    tokenizer=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    full = SimulatedTriageDataset(
        n=n,
        image_size=image_size,
        seed=seed,
        augment=False,
        use_roberta_tokenizer=use_roberta_tokenizer,
        tokenizer=tokenizer,
    )
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    gen = torch.Generator().manual_seed(seed)
    train_s, val_s, test_s = random_split(full, [n_train, n_val, n_test], generator=gen)

    train_ds = SimulatedTriageDataset(
        n=n,
        image_size=image_size,
        seed=seed,
        augment=True,
        use_roberta_tokenizer=use_roberta_tokenizer,
        tokenizer=tokenizer,
    )
    # Keep the same indices as train_s but with augmentation enabled.
    train_subset = torch.utils.data.Subset(train_ds, train_s.indices)

    def _loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return _loader(train_subset, True), _loader(val_s, False), _loader(test_s, False)
