# SkinSight AI — PAD-UFES-20 + ISIC Combined Training

This is the upgraded build of the new skin-cancer/skin-lesion project.

It deliberately does **not** reuse the previous project's HAM10000 + SD-198 + EfficientNetB0 setup.

## Final data strategy

### Primary source: PAD-UFES-20
Used because it contains smartphone clinical photographs and is therefore directly relevant to the project's mobile-camera use case.

### Secondary source: ISIC 2019
Used to add a much larger dermoscopic lesion source for overlapping lesion classes.

The sources are NOT blindly merged. Each image is stored in a combined manifest with:
- source
- mapped class
- patient/group identifier where available
- original image ID
- image path

The sampler balances `source + class`, helping prevent ISIC from overwhelming the smaller smartphone dataset.

## Shared six-class label space

| Final label | Meaning | PAD-UFES-20 | ISIC 2019 |
|---|---|---|---|
| ACK | Actinic Keratosis | ACK | AK |
| BCC | Basal Cell Carcinoma | BCC | BCC |
| MEL | Melanoma | MEL | MEL |
| NEV | Melanocytic Nevus | NEV | NV |
| SCC | Squamous Cell Carcinoma | SCC | SCC |
| SEK | Seborrheic / benign keratosis | SEK | BKL |

ISIC classes outside this shared six-class target (for example dermatofibroma and vascular lesions) are intentionally excluded.

## Folder layout after downloading datasets

```text
SkinSight_AI/
  data/
    pad/
      metadata.csv
      images/
        ...
    isic2019/
      ground_truth.csv
      metadata.csv        # optional but strongly preferred
      images/
        ...
  ml/
    prepare_combined.py
    train.py
```

## 1. Create the Python environment

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Build the combined manifest

Adjust filenames if your downloaded CSVs have different names:

```powershell
python ml\prepare_combined.py `
  --pad-csv data\pad\metadata.csv `
  --pad-images data\pad\images `
  --isic-gt data\isic2019\ground_truth.csv `
  --isic-images data\isic2019\images `
  --isic-meta data\isic2019\metadata.csv
```

This creates:

```text
data/combined_manifest.csv
```

Before training, the command prints counts by dataset source and class. Check those counts.

## 3. Train one combined model

```powershell
python ml\train.py --manifest data\combined_manifest.csv --epochs 20 --batch-size 16
```

Output:

```text
models/skinsight_convnext_tiny.pt
models/metrics.json
```

The checkpoint metadata records:

```text
dataset = PAD-UFES-20 + ISIC-2019
architecture = convnext_tiny
```

## 4. Start the application

```powershell
python run.py
```

Open:

```text
http://127.0.0.1:8000
```

For another device on the same network:

```powershell
python run.py --host 0.0.0.0
```

Then use the computer's IPv4 address from `ipconfig`.

## Why this is safer than simply combining folders

PAD-UFES-20 clinical smartphone photographs and ISIC dermoscopy images have very different image styles. A model can learn those source differences instead of genuine disease features.

This build reduces that risk by:
1. preserving the source of every image,
2. balancing source + class during sampling,
3. using patient/group-aware train/validation splitting where patient IDs are available,
4. preventing known groups from appearing in both training and validation,
5. reporting validation balanced accuracy rather than plain accuracy only.

For a research-grade final evaluation, we should additionally create:
- a PAD-only holdout test set,
- an ISIC-only holdout test set,
- confusion matrices and per-class sensitivity/specificity,
- external validation on images never used during development.

## Medical limitation

This project is a research/educational screening tool. A photo classifier cannot determine clinical Stage I, II, III, or IV skin cancer from an image alone.

Staging can require pathology/biopsy, Breslow thickness or tumor depth, ulceration, lymph-node status, imaging, and evidence of distant spread.

The application therefore provides lesion-class probabilities and next-step guidance without fabricating a clinical stage.
