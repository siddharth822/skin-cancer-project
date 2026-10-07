"""Evaluate new external data; never treat the old reported test as a new test."""
import argparse
import json
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import inference
from train import report_frame
from external_data import audit_external


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve previous evaluation reports; choose a new output')
    inference.CHECKPOINT = Path(args.checkpoint)
    predictor = inference.Predictor()
    if not predictor.ready:
        raise FileNotFoundError('Checkpoint missing')
    calibration_reference = None
    if predictor.meta.get("calibration"):
        calibration_reference = Path(args.checkpoint).with_suffix(".calibration.csv")
        if not calibration_reference.exists():
            raise FileNotFoundError("Calibrated model requires its calibration identity CSV to exclude that cohort from testing")
    frame, audit = audit_external(args.manifest, args.reference_dir, calibration_reference)
    # report_frame applies temperature to logits when the checkpoint supplies it.
    model = predictor.model
    if predictor.temperature != 1:
        class ScaledModel(torch.nn.Module):
            def __init__(self, model, temperature):
                super().__init__(); self.model = model; self.temperature = temperature
            def forward(self, x):
                return self.model(x) / self.temperature
        model = ScaledModel(model, predictor.temperature)
    report = report_frame(model, frame, predictor.image_size, 32, 0, predictor.device)
    report['external_audit'] = audit
    report['evaluation_note'] = 'External research evaluation. Identity checks alone do not certify patient independence or clinical validity.'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print('External balanced accuracy:', report['balanced_accuracy'])


if __name__ == '__main__': main()
