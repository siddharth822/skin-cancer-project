"""Frozen ImageNet feature caching for a fast CPU classifier-head baseline."""
import torch
from torch.utils.data import DataLoader, TensorDataset
from dataset import ManifestDataset


@torch.no_grad()
def cache_features(model, frame, transform, device, batch_size, workers):
    model.eval()
    loader = DataLoader(ManifestDataset(frame, transform), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    features, labels = [], []
    for i, (images, targets) in enumerate(loader, 1):
        values = model.avgpool(model.features(images.to(device))).flatten(1)
        features.append(values.cpu()); labels.append(targets)
        if i % 100 == 0 or i == len(loader):
            print(f'Frozen feature extraction {i}/{len(loader)}', flush=True)
    return TensorDataset(torch.cat(features), torch.cat(labels))
