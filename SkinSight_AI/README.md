# SkinSight AI

Educational skin-lesion classification using **MobileNetV3-Small**. Six classes:
ACK, BCC, MEL, NEV, SCC, SEK. ISIC AK maps to ACK, NV to NEV, BKL to SEK.
BKL is broader than PAD's seborrheic keratosis, so this mapping is approximate.

## Easiest GPU training workflow

Import `notebooks/train_kaggle.ipynb` into Kaggle. Enable an available GPU and
Internet, then Run All. The notebook bundles this code, downloads the public
PAD and resized ISIC datasets plus ground truth, validates matching, trains,
and exports `skinsight_trained_app.zip`. GPU/account access and full-data downloads
have not been verified here. Follow dataset licenses and attribution requirements.

## Windows local workflow

Run these commands from `SkinSight_AI` (Python 3.12 recommended):

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python ml\prepare_combined.py --pad-meta "D:\SkinCancerData\PAD-UFES-20\extracted\metadata.csv" --pad-root "D:\SkinCancerData\PAD-UFES-20\extracted" --isic-gt "D:\SkinCancerData\ISIC-2019\ISIC_2019_Training_GroundTruth.csv" --isic-root "D:\SkinCancerData\ISIC-2019\extracted\train-image" --out data\combined_manifest.csv
.\.venv\Scripts\python ml\train.py --manifest data\combined_manifest.csv --epochs 10 --batch-size 16 --size 160 --fine-tune --device auto --output-dir models
.\.venv\Scripts\python run.py
```

PAD image folders are scanned recursively. Missing images cause an error instead
of a partial manifest. CPU training on a low-memory laptop is likely slow.
`--device auto` selects CUDA if available; `--device cuda` requires a CUDA GPU.
Pretrained weights download from PyTorch. Without `--fine-tune`, the backbone and
its BatchNorm statistics stay frozen. `--from-scratch` requires `--fine-tune`.

Outputs: `models/skinsight_mobilenetv3_small.pt`, `metrics.json`, `train_split.csv`,
`validation_split.csv`. Reports include per-source validation metrics.

## Connect trained weights

Copy the trained checkpoint to `SkinSight_AI/models/skinsight_mobilenetv3_small.pt`
and restart the app. Alternatively set `SKINSIGHT_MODEL_PATH` to an absolute path.
Only use trusted checkpoints; loading uses `weights_only=True`.
Without a checkpoint `/api/health` reports `model_ready=false` and predictions
return HTTP 503. Synthetic test weights must never be deployed.

## Evaluation limits

PAD groups by real patient IDs. ISIC ground truth has no patient IDs; official
metadata supplies lesion groups where present. Lesion or image grouping **does not
establish patient-level separation**. The resized mirror's `dummy_*` IDs are not
genuine patient IDs. Validation selects the checkpoint; a separate internal test
partition evaluates it afterward. Genuine ISIC patient metadata, probability
calibration, and external smartphone testing remain needed. Source/class balancing cannot
eliminate clinical-versus-dermoscopic domain shift. The 224px mirror is a practical
baseline; original ISIC images contain more detail.

This app is an educational research tool, not a diagnosis or clinical staging
system. No real-data model has yet been trained or clinically validated here.

## Tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Regression tests use artificial images to check software behavior only.

## Independent holdout and grouping

Training now creates a fixed, source/class-stratified **train/validation/test**
partition (approximately 60/20/20). Linked patient IDs, lesion IDs, image IDs,
and exact-file SHA-256 duplicates stay in a single partition. Each partition must
contain every source/class represented in the manifest; otherwise training fails
with a diagnostic rather than silently reporting incomplete results.

Supply `--isic-meta path/to/ISIC_2019_Training_Metadata.csv` to preparation to
use authoritative ISIC lesion IDs and genuine patient IDs where available.
The Kaggle notebook downloads the official metadata automatically. Dummy patient
IDs are ignored. `split_summary.json` records grouping coverage; missing IDs remain
explicit image-level groups. File hashes detect exact duplicates, not re-encoded
or cropped copies. Lesion separation still does not prove patient separation.

The test partition is never used to select checkpoints. After training, the best
validation checkpoint is loaded and test metrics are computed once. Outputs now
also include `test_split.csv`, `split_summary.json`, and `test_metrics.json`.
Overall and per-source test reports contain confusion matrices, per-class
sensitivity/specificity, one-vs-rest ROC AUC (null when undefined), and calibration
error. PAD-only test metrics are the relevant internal smartphone-photo measure.
External smartphone performance remains unknown until separately evaluated.

Use a fresh output directory for another run. Reusing test feedback to tune a
model contaminates the test set even when filenames are different. Existing
checkpoints/test results will not be overwritten automatically.

## Fast CPU baseline

Use `--cache-features` without `--fine-tune` or `--from-scratch` to extract frozen
pretrained MobileNet features once and train only its classifier. For example:

```powershell
.\.venv\Scripts\python ml\train.py --manifest data\combined_manifest.csv --epochs 5 --batch-size 64 --size 160 --cache-features --device cpu --output-dir models_baseline
```

This mode uses deterministic resize/normalization without training augmentation.
It is a quick baseline, not full-model fine-tuning. Cached features are kept in RAM;
validation/test partitioning and checkpoint selection rules remain unchanged.
The resulting checkpoint still loads through the same full MobileNet predictor.
