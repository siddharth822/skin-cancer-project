from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

class ManifestDataset(Dataset):
    """
    Dataset backed by data/combined_manifest.csv.

    Required columns:
      image_path,label,patient_id,source,original_id
    """
    def __init__(self, frame, transform=None):
        self.df = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(row["image_path"])
        image = Image.open(path).convert("RGB")
        label = str(row["label"]).strip().upper()
        if label not in CLASS_TO_IDX:
            raise ValueError(f"Unsupported label: {label}")
        y = CLASS_TO_IDX[label]

        if self.transform:
            import numpy as np
            image = self.transform(image=np.asarray(image))["image"]

        return image, y
