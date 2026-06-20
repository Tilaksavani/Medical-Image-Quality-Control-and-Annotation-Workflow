from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.preprocess import preprocess_for_model


LABEL_TO_ID: Dict[str, int] = {
    "normal": 0,
    "low_contrast": 1,
    "artifact": 2,
}

ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


class MedicalImageDataset(Dataset):
    """Dataset backed by an annotation CSV with file_name and label columns."""

    def __init__(self, annotations_csv: str | Path, image_dir: str | Path):
        self.annotations = pd.read_csv(annotations_csv)
        required = {"file_name", "label"}
        missing = required.difference(set(self.annotations.columns))
        if missing:
            raise ValueError(f"Missing columns in annotation CSV: {sorted(missing)}")

        self.image_dir = Path(image_dir)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.annotations.iloc[index]
        image_path = self.image_dir / row["file_name"]
        image = preprocess_for_model(image_path)
        label = LABEL_TO_ID[str(row["label"])]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
