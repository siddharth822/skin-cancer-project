"""Train on training data, select on validation, evaluate held-out test once."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from dataset import ManifestDataset, CLASS_NAMES
from model import create_model
from splits import linked_groups, split_three_way
from evaluation import summarize


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def tfms(train, size):
    items = [A.Resize(size, size)]
    if train:
        items += [A.HorizontalFlip(p=.5), A.Rotate(limit=20, p=.3),
                  A.RandomBrightnessContrast(p=.25), A.HueSaturationValue(p=.15)]
    return A.Compose(items + [A.Normalize(), ToTensorV2()])


def split_df(df):
    """Compatibility helper for callers needing only a train/validation partition."""
    df = df[df.label.isin(CLASS_NAMES)].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError('Manifest contains no supported lesion classes')
    df['_group'] = linked_groups(df)
    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(df, groups=df['_group']))
    return df.iloc[a].copy(), df.iloc[b].copy()


def sampler(df):
    keys = df.source.astype(str) + '::' + df.label.astype(str)
    counts = keys.value_counts().to_dict()
    return WeightedRandomSampler([1 / counts[key] for key in keys], len(keys), replacement=True)


@torch.no_grad()
def collect(model, loader, device):
    model.eval(); labels = []; probabilities = []; loss_sum = 0.
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(nn.functional.cross_entropy(logits, y)) * len(y)
        labels.extend(y.cpu().tolist())
        probabilities.extend(torch.softmax(logits, 1).cpu().tolist())
    return loss_sum / len(loader.dataset), labels, probabilities


def evaluate(model, loader, device):
    loss, labels, probabilities = collect(model, loader, device)
    report = summarize(labels, probabilities)
    return loss, report['balanced_accuracy'], report['classification_report']


def report_frame(model, frame, size, batch_size, workers, device):
    loader = DataLoader(ManifestDataset(frame, tfms(False, size)), batch_size=batch_size,
                        shuffle=False, num_workers=workers)
    loss, labels, probabilities = collect(model, loader, device)
    report = summarize(labels, probabilities); report['loss'] = loss
    report['by_source'] = {}
    sources = frame.source.to_numpy()
    for source in sorted(frame.source.unique()):
        selected = np.flatnonzero(sources == source)
        report['by_source'][source] = summarize(np.asarray(labels)[selected], np.asarray(probabilities)[selected])
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='data/combined_manifest.csv')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--size', type=int, default=160)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--output-dir', default='models')
    p.add_argument('--from-scratch', action='store_true')
    p.add_argument('--fine-tune', action='store_true')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args(); seed_all()
    if a.epochs < 1 or a.batch_size < 2 or a.size < 32 or a.workers < 0 or a.lr <= 0:
        p.error('epochs/lr must be positive, batch-size >= 2, size >= 32, workers >= 0')
    if a.from_scratch and not a.fine_tune:
        p.error('--from-scratch requires --fine-tune so random features are not frozen')
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested, but no CUDA GPU is available')
    device = torch.device('cuda' if a.device == 'auto' and torch.cuda.is_available() else
                          ('cpu' if a.device == 'auto' else a.device))
    out = Path(a.output_dir); checkpoint = out / 'skinsight_mobilenetv3_small.pt'
    if checkpoint.exists() or (out / 'test_metrics.json').exists():
        raise FileExistsError('Training output already exists; use a new --output-dir to preserve prior results')
    df = pd.read_csv(a.manifest)
    required = {'image_path', 'label', 'source', 'original_id'}
    if required - set(df.columns):
        raise ValueError(f'Manifest lacks columns: {sorted(required-set(df.columns))}')
    if not set(df.label).issubset(CLASS_NAMES):
        raise ValueError('Manifest contains unsupported labels')
    if 'image_sha256' not in df or df.image_sha256.isna().any():
        raise ValueError('Rebuild the manifest to include image_sha256 for duplicate-aware splitting')
    tr, va, test = split_three_way(df)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [('train', tr), ('validation', va), ('test', test)]:
        part.to_csv(out / f'{name}_split.csv', index=False)
        print(name, len(part)); print(part.groupby(['source', 'label']).size().unstack(fill_value=0))
    grouping = df.groupby(['source', 'grouping_level']).size().to_dict() if 'grouping_level' in df else {}
    split_info = {
        'seed': 42, 'manifest_sha256': hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
        'counts': {'train': len(tr), 'validation': len(va), 'test': len(test)},
        'grouping_coverage': {f'{source}::{level}': int(count) for (source, level), count in grouping.items()},
        'limitations': 'Linked patient, lesion and exact-file hashes stay together. Image/lesion groups do not prove patient separation. Re-encoded near-duplicates and unseen external data need further evaluation.',
    }
    (out / 'split_summary.json').write_text(json.dumps(split_info, indent=2))
    training = DataLoader(ManifestDataset(tr, tfms(True, a.size)), batch_size=a.batch_size,
                          sampler=sampler(tr), num_workers=a.workers, pin_memory=device.type == 'cuda',
                          drop_last=len(tr) % a.batch_size == 1)
    validation = DataLoader(ManifestDataset(va, tfms(False, a.size)), batch_size=a.batch_size,
                            shuffle=False, num_workers=a.workers)
    model = create_model(pretrained=not a.from_scratch, freeze_backbone=not a.fine_tune).to(device)
    optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=a.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    best = -1.; history = []; best_epoch = None
    print('Device:', device, 'Architecture: mobilenet_v3_small', flush=True)
    for epoch in range(1, a.epochs + 1):
        start = time.time(); model.train(); run = 0.
        if not a.fine_tune:
            model.features.eval()
        for i, (x, y) in enumerate(training, 1):
            x, y = x.to(device), y.to(device); optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = criterion(logits, y); loss.backward(); optimizer.step()
            run += float(loss.detach()) * len(y)
            if i % 100 == 0 or i == len(training):
                print(f'epoch {epoch}/{a.epochs}, batch {i}/{len(training)}', flush=True)
        vl, score, report = evaluate(model, validation, device)
        history.append({'epoch': epoch, 'train_loss': run / len(tr), 'validation_loss': vl,
                        'validation_balanced_accuracy': score, 'seconds': time.time() - start})
        print(history[-1], flush=True)
        if score > best:
            best = score; best_epoch = epoch
            torch.save({'model_state': model.state_dict(), 'class_names': CLASS_NAMES,
                        'architecture': 'mobilenet_v3_small', 'image_size': a.size,
                        'balanced_accuracy': best, 'validation_report': report,
                        'dataset': 'PAD-UFES-20 + ISIC-2019',
                        'training_mode': 'from_scratch' if a.from_scratch else ('imagenet_finetune' if a.fine_tune else 'imagenet_head_only')}, checkpoint)
    # Only after selection, reload the best checkpoint and evaluate test data once.
    best_checkpoint = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(best_checkpoint['model_state'])
    validation_report = report_frame(model, va, a.size, a.batch_size, a.workers, device)
    test_report = report_frame(model, test, a.size, a.batch_size, a.workers, device)
    test_report['evaluation_note'] = 'Held out from fitting and checkpoint selection. Reusing this test set to tune the model invalidates independent-test claims. Patient independence depends on grouping coverage.'
    test_report['split_summary'] = split_info
    (out / 'test_metrics.json').write_text(json.dumps(test_report, indent=2))
    (out / 'metrics.json').write_text(json.dumps({'best_epoch': best_epoch, 'best_balanced_accuracy': best,
                                                'history': history, 'validation': validation_report}, indent=2))
    print('Saved:', checkpoint, 'and test_metrics.json', flush=True)


if __name__ == '__main__':
    main()
