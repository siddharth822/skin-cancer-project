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
system. A real-data frozen-feature baseline has now been trained; see reports/baseline_v1/RESULTS.md. It has not been clinically validated.

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

## Fine-tuning without reusing the reported test

Warm-start from the baseline and select using PAD smartphone validation:

```powershell
.\.venv\Scripts\python ml\train.py --manifest data\combined_manifest.csv --init-checkpoint models\skinsight_mobilenetv3_small.pt --fine-tune --epochs 5 --lr 0.00003 --batch-size 32 --size 160 --selection-source PAD-UFES-20 --skip-test --output-dir models_finetune
```

`--skip-test` never loads test images or computes test metrics. It still preserves
split identity records for auditing. The warm-start checkpoint is evaluated as epoch
zero and retained if no fine-tuned epoch improves the chosen validation score.
Source-specific selection scores are labeled; they must not be confused with the
combined validation score or old held-out scores. Full-model training is slower
than cached classifier-head training, particularly on a CPU.

## New external evaluation and calibration

Provide a new, independently collected labeled manifest with columns
`image_path,label,source,original_id,image_sha256,patient_id,lesion_id`. Labels must
come from authoritative ground truth and map to the supported six classes; do not
invent labels. SHA-256 must match each actual image. Missing patient identifiers
limit independence claims; genuine patient IDs are required for calibration.

```powershell
.\.venv\Scripts\python ml\evaluate_external.py --manifest external_test.csv --checkpoint models_finetune\skinsight_mobilenetv3_small.pt --reference-dir models_baseline --output external_report.json
.\.venv\Scripts\python ml\calibrate.py --manifest new_calibration.csv --checkpoint models_finetune\skinsight_mobilenetv3_small.pt --reference-dir models_baseline --output-checkpoint models_finetune\calibrated.pt
```

Both tools reject overlap with the recorded training, validation, and reported test
identities and exact file hashes. Aliased identities and re-encoded images still
need a cohort/provenance review. Calibration uses a **new cohort**, not the old
reported test. Temperature scaling changes model scores but not class ordering or
accuracy. NLL improvement on calibration data is not proof of performance on a new
test. Clinical confidence remains unvalidated.

The calibrated checkpoint has a companion `.calibration.csv` identity file. Keep it
private and alongside the checkpoint when evaluating a new external test: the
external evaluation command requires it and also rejects calibration/test overlap.
Do not upload identity CSVs or patient images to GitHub. Only aggregate reports may
be published. The app accepts positive finite temperature metadata automatically.

A candidate independently collected clinical cohort is Stanford's DDI (656 images,
570 patients, biopsy-grounded diagnoses). Its official documentation requires each
user to register and agree to its research-use terms. Access instructions are at
https://ddi-dataset.github.io/#access and the official portal is linked there.
DDI was **not downloaded or evaluated** in this workspace. Its supported categories,
patient identifiers, and permitted use must be checked before evaluation. If patient
identifiers are unavailable, do not randomly split its images into supposedly
patient-independent calibration and test sets.

## Upload quality checks

The API rejects tiny images, blank/extremely exposed images, grayscale inputs and obvious flat-color graphics before classification (HTTP 422, no prediction). Changing images or receiving an error clears the old result. These conservative heuristics are not a trained skin/lesion detector: unrelated natural photos may still pass, and legitimate images may be rejected. Do not treat acceptance as proof that a lesion is present. A validated semantic detector needs representative skin and non-skin training and evaluation data.

## Accounts and login

On Windows, double-click `start_windows.bat` from the extracted app folder, or use the existing PowerShell setup commands. Open the local app, choose **Create an account**, register a username and a password of at least 12 characters, then log in. Logout revokes the session. Accounts persist in `data/accounts.sqlite3` on the machine hosting this app; separate computers have separate accounts. Keep this database private and outside shared ZIPs/Git. Uploaded photos are processed in memory; screening report summaries are saved privately for downloads.

Passwords use salted PBKDF2-SHA256 hashes; sessions are random server-side records with an eight-hour expiry. Login attempts are rate limited, and POST operations require CSRF tokens. This is a local demo with verification of report email addresses and no password recovery. For HTTPS hosting set `SKINSIGHT_SECURE_COOKIES=1`; internet deployment requires its own operational review. The health endpoint is public, but the screening page and prediction API require login.

Cancer stage information is educational. The app never assigns a clinical stage from an image. For melanoma it explains stages 0–IV; for BCC/SCC it explains the different assessment requirements. Stage determination needs confirmed diagnosis and clinical/pathology findings.

## Download reports and email through Gmail

Every successful analysis creates an account-owned PDF screening report, with the image-model result, class scores, doctor guidance and stage limitations. Choose **Download PDF report** below the result. The PDF and email do not attach your uploaded image. Report summaries persist privately in the account database; other accounts cannot download them. Rejected images create no report and send no report email.

To enable sending on your Windows computer:

1. Use a Gmail account as the application's sender. Enable Google 2-Step Verification, then create an **App Password**: https://support.google.com/accounts/answer/185833. Some organizational/managed accounts may not offer App Passwords. Never use or share your normal Gmail password.
2. Stop an existing app with Ctrl+C. Double-click **start_with_gmail.bat**. Enter the sender Gmail address and its App Password at the masked local prompt. The launcher keeps the password only in the process environment for this run, not in a saved file. Gmail setup must be supplied again when restarting this launcher.
3. Register with your report email, then log in (username or email). Open **Email settings → Send verification code**, check your Gmail inbox/spam, and enter the code. Existing accounts can save an email there without re-registering.
4. Once verified, every successful future analysis automatically emails the screening report PDF and doctor guidance. Check the email status shown below the result. Downloading works even if sending fails.

Server configuration uses `SKINSIGHT_SMTP_HOST=smtp.gmail.com`, `SKINSIGHT_SMTP_PORT=465`, `SKINSIGHT_SMTP_USER`, `SKINSIGHT_SMTP_PASSWORD` (App Password), and `SKINSIGHT_MAIL_FROM`. SMTP uses verified TLS; plain SMTP is never used for credentials. Provider acceptance does not guarantee inbox delivery. Email sending is best effort, not a durable queue; restarting during delivery can interrupt it. No real Gmail delivery has been tested without configured credentials. Codes expire in ten minutes, permit five guesses, and sending codes is limited to once per minute per account.

The sending account is separate from each user's recipient email. Keep local databases, reports and sender credentials private. Shared ZIPs contain none of those. This is educational screening, not diagnosis, prescribed treatment or clinical staging.
