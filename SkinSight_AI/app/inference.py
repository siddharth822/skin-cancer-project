from pathlib import Path
import io
import os
import math
import numpy as np
from PIL import Image
import torch
from torchvision.models import mobilenet_v3_small
from torch import nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.environ.get("SKINSIGHT_MODEL_PATH", str(PROJECT_ROOT / "models/skinsight_mobilenetv3_small.pt")))

class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None
        self.image_size = 160
        self.temperature = 1.0
        self.meta = {}
        self._load()

    def _load(self):
        if not CHECKPOINT.exists():
            return
        ckpt = torch.load(CHECKPOINT, map_location=self.device, weights_only=True)
        self.temperature = float(ckpt.get("temperature", 1.0))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Checkpoint temperature must be finite and positive")
        self.class_names = ckpt["class_names"]
        self.image_size = int(ckpt.get("image_size", 160))
        self.meta = {
            "architecture": ckpt.get("architecture", "mobilenet_v3_small"),
            "dataset": ckpt.get("dataset", "PAD-UFES-20"),
            "balanced_accuracy": ckpt.get("balanced_accuracy"),
            "training_mode": ckpt.get("training_mode"),
            "temperature": self.temperature,
            "selection_source": ckpt.get("selection_source", "combined"),
            "calibration": ckpt.get("calibration"),
        }
        if self.meta["architecture"] != "mobilenet_v3_small":
            raise ValueError("Expected a MobileNetV3-Small checkpoint produced by ml/train.py")
        if set(self.class_names) != {"ACK", "BCC", "MEL", "NEV", "SCC", "SEK"} or len(self.class_names) != 6:
            raise ValueError("Checkpoint must contain the six supported lesion classes")
        self.model = mobilenet_v3_small(weights=None)
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, len(self.class_names)
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        self.transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(),
            ToTensorV2(),
        ])

    @property
    def ready(self):
        return self.model is not None

    @torch.no_grad()
    def predict_bytes(self, raw: bytes):
        if not self.ready:
            raise RuntimeError("Model is not trained yet. Train it first with ml/train.py.")

        image = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(image)
        x = self.transform(image=arr)["image"].unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.softmax(logits / self.temperature, dim=1)[0].cpu().numpy()

        order = np.argsort(probs)[::-1]
        ranked = [
            {"label": self.class_names[i], "probability": round(float(probs[i]) * 100, 2)}
            for i in order[:3]
        ]
        return {
            "label": ranked[0]["label"],
            "confidence": ranked[0]["probability"],
            "top_predictions": ranked,
        }
