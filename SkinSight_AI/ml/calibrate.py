"""Temperature scaling on NEW calibration data, separate from reported evaluation."""
import argparse
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import inference
from dataset import ManifestDataset
from train import tfms
from external_data import audit_external


def fit_temperature(logits, labels):
    """One positive scalar; preserves class ordering and cannot increase accuracy."""
    logits = logits.detach().cpu().double(); labels = labels.detach().cpu()
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=.1, max_iter=100, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.clamp(-3, 3).exp(), labels)
        loss.backward(); return loss
    before = float(torch.nn.functional.cross_entropy(logits, labels))
    optimizer.step(closure)
    temperature = float(log_temperature.detach().clamp(-3, 3).exp())
    after = float(torch.nn.functional.cross_entropy(logits / temperature, labels))
    if not torch.isfinite(torch.tensor(after)) or after > before + 1e-8:
        return 1., before, before
    return temperature, before, after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, help='New calibration cohort, not an already reported test')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference-dir', required=True, help='Run with train/validation/test split CSVs')
    parser.add_argument('--output-checkpoint', required=True)
    args = parser.parse_args()
    output = Path(args.output_checkpoint)
    if output.exists():
        raise FileExistsError('Choose a new calibration output; existing checkpoints are preserved')
    frame, audit = audit_external(args.manifest, args.reference_dir)
    if 'patient_id' not in frame or frame.patient_id.isna().any():
        raise ValueError('Calibration requires real patient groups to separate calibration from subsequent testing')
    patients = frame.patient_id.astype(str).str.strip()
    if patients.eq('').any() or patients.str.lower().str.startswith('dummy').any():
        raise ValueError('Calibration requires genuine patient IDs, not empty or dummy IDs')
    inference.CHECKPOINT = Path(args.checkpoint)
    predictor = inference.Predictor()
    if not predictor.ready:
        raise FileNotFoundError('Checkpoint missing')
    if predictor.meta.get('calibration'):
        raise ValueError('Start from an uncalibrated checkpoint; repeated calibration would obscure cohort provenance')
    logits, labels = [], []
    loader = DataLoader(ManifestDataset(frame, tfms(False, predictor.image_size)), batch_size=32)
    with torch.no_grad():
        for images, targets in loader:
            logits.append(predictor.model(images.to(predictor.device)).cpu())
            labels.append(targets)
    temperature, before, after = fit_temperature(torch.cat(logits), torch.cat(labels))
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    checkpoint['temperature'] = temperature
    checkpoint['calibration'] = {'method': 'temperature_scaling', 'audit': audit,
                                 'negative_log_likelihood_before': before,
                                 'negative_log_likelihood_after': after,
                                 'clinical_validation': False,
                                 'note': 'Fit on a calibration cohort; improvement on this same cohort does not prove external calibration. Never evaluate using the calibration data itself.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    # Preserve cohort identity records so a later external test can also exclude them.
    frame.to_csv(output.with_suffix('.calibration.csv'), index=False)
    print(json.dumps({'temperature': temperature, 'calibration_nll_before': before, 'calibration_nll_after': after}))


if __name__ == '__main__': main()
