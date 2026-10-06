import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

NUM_CLASSES = 6
CLASS_NAMES = ["ACK","BCC","MEL","NEV","SCC","SEK"]

def create_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True):
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model
