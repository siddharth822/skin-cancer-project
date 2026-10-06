from pathlib import Path
import io, os, numpy as np, torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models import mobilenet_v3_small
from torch import nn

D=Path(r"D:\SkinCancerData\models\skinsight_mobilenetv3_small.pt")
L=Path("models/skinsight_mobilenetv3_small.pt")
CHECKPOINT=Path(os.environ.get("SKINSIGHT_MODEL_PATH",str(D if D.exists() else L)))

class Predictor:
    def __init__(self):
        self.device=torch.device("cpu"); self.model=None; self.class_names=None; self.image_size=160; self.meta={}
        self._load()

    def _load(self):
        if not CHECKPOINT.exists(): return
        c=torch.load(CHECKPOINT,map_location="cpu",weights_only=False)
        self.class_names=c["class_names"]; self.image_size=int(c.get("image_size",160))
        self.meta={"architecture":"mobilenet_v3_small","dataset":c.get("dataset"),"balanced_accuracy":c.get("balanced_accuracy")}
        self.model=mobilenet_v3_small(weights=None)
        self.model.classifier[-1]=nn.Linear(self.model.classifier[-1].in_features,len(self.class_names))
        self.model.load_state_dict(c["model_state"]); self.model.eval()
        self.transform=A.Compose([A.Resize(self.image_size,self.image_size),A.Normalize(),ToTensorV2()])

    @property
    def ready(self): return self.model is not None

    @torch.no_grad()
    def predict_bytes(self,raw):
        if not self.ready: raise RuntimeError(f"Model not found: {CHECKPOINT}")
        arr=np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        x=self.transform(image=arr)["image"].unsqueeze(0)
        p=torch.softmax(self.model(x),1)[0].numpy()
        order=np.argsort(p)[::-1]
        top=[{"label":self.class_names[i],"probability":round(float(p[i])*100,2)} for i in order[:3]]
        return {"label":top[0]["label"],"confidence":top[0]["probability"],"top_predictions":top}
