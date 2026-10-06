from pathlib import Path
import io
import numpy as np
from PIL import Image
import torch
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

CHECKPOINT = Path("models/skinsight_convnext_tiny.pt")

class Predictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.class_names = None
        self.image_size = 224
        self.meta = {}
        self._load()

    def _load(self):
        if not CHECKPOINT.exists():
            return
        ckpt = torch.load(CHECKPOINT, map_location=self.device, weights_only=False)
        self.class_names = ckpt["class_names"]
        self.image_size = int(ckpt.get("image_size", 224))
        self.meta = {
            "architecture": ckpt.get("architecture", "convnext_tiny"),
            "dataset": ckpt.get("dataset", "PAD-UFES-20"),
            "balanced_accuracy": ckpt.get("balanced_accuracy"),
            "training_mode": ckpt.get("training_mode"),
        }
        self.model = timm.create_model(
            ckpt.get("architecture", "convnext_tiny"),
            pretrained=False,
            num_classes=len(self.class_names),
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
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

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
