import argparse
from pathlib import Path
import pandas as pd

TARGET_CLASSES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]

PAD_ALIASES = {
    "ACK": "ACK",
    "AK": "ACK",
    "ACTINIC KERATOSIS": "ACK",
    "BCC": "BCC",
    "BASAL CELL CARCINOMA": "BCC",
    "MEL": "MEL",
    "MELANOMA": "MEL",
    "NEV": "NEV",
    "NEVUS": "NEV",
    "SCC": "SCC",
    "SQUAMOUS CELL CARCINOMA": "SCC",
    "SEK": "SEK",
    "SK": "SEK",
    "SEBORRHEIC KERATOSIS": "SEK",
}

ISIC_MAP = {
    "MEL": "MEL",
    "NV": "NEV",
    "BCC": "BCC",
    "AK": "ACK",
    "BKL": "SEK",
    "SCC": "SCC",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

def find_column(df, candidates, required=True):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lookup:
            return lookup[c.lower()]
    if required:
        raise ValueError(f"Could not find any of {candidates}. Found columns: {list(df.columns)}")
    return None

def build_image_index(root: Path):
    index = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in VALID_EXTS:
            index.setdefault(p.stem, p.resolve())
            index.setdefault(p.name, p.resolve())
    return index

def normalize_pad(v):
    return PAD_ALIASES.get(str(v).strip().upper())

def build_pad_manifest(metadata_csv: Path, pad_root: Path):
    df = pd.read_csv(metadata_csv)
    image_col = find_column(df, ["img_id", "image", "image_id", "filename", "file_name"])
    label_col = find_column(df, ["diagnostic", "diagnosis", "label", "class"])
    patient_col = find_column(df, ["patient_id", "patient", "patientid"], required=False)

    image_index = build_image_index(pad_root)
    rows = []
    missing = 0

    for _, row in df.iterrows():
        label = normalize_pad(row[label_col])
        if label not in TARGET_CLASSES:
            continue

        raw = str(row[image_col]).strip()
        path = image_index.get(raw) or image_index.get(Path(raw).stem)

        if path is None:
            missing += 1
            continue

        patient_id = ""
        if patient_col and pd.notna(row[patient_col]):
            patient_id = f"PAD::{row[patient_col]}"

        rows.append({
            "image_path": str(path),
            "label": label,
            "patient_id": patient_id,
            "source": "PAD-UFES-20",
            "original_id": Path(raw).stem,
        })

    out = pd.DataFrame(rows)
    print(f"PAD matched images: {len(out)} | missing: {missing}")
    return out

def build_isic_manifest(gt_csv: Path, image_root: Path):
    gt = pd.read_csv(gt_csv)
    image_col = find_column(gt, ["image", "image_id", "isic_id"])
    image_index = build_image_index(image_root)

    rows = []
    missing = 0
    skipped = 0

    for _, row in gt.iterrows():
        positives = []
        for src_label in ISIC_MAP:
            if src_label in gt.columns:
                try:
                    if float(row[src_label]) >= 0.5:
                        positives.append(src_label)
                except Exception:
                    pass

        if len(positives) != 1:
            skipped += 1
            continue

        src_label = positives[0]
        final_label = ISIC_MAP[src_label]
        image_id = str(row[image_col]).strip()
        path = image_index.get(image_id) or image_index.get(Path(image_id).stem)

        if path is None:
            missing += 1
            continue

        rows.append({
            "image_path": str(path),
            "label": final_label,
            "patient_id": "",
            "source": "ISIC-2019",
            "original_id": Path(image_id).stem,
        })

    out = pd.DataFrame(rows)
    print(f"ISIC matched images: {len(out)} | missing: {missing} | unsupported/skipped: {skipped}")
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pad-meta", required=True)
    p.add_argument("--pad-root", required=True)
    p.add_argument("--isic-gt", required=True)
    p.add_argument("--isic-root", required=True)
    p.add_argument("--out", default=r"D:\SkinCancerData\combined\combined_manifest.csv")
    args = p.parse_args()

    pad = build_pad_manifest(Path(args.pad_meta), Path(args.pad_root))
    isic = build_isic_manifest(Path(args.isic_gt), Path(args.isic_root))

    combined = pd.concat([pad, isic], ignore_index=True)
    combined = combined.drop_duplicates(subset=["source", "original_id"]).reset_index(drop=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)

    print("\n=== FINAL COMBINED DATASET ===")
    print(combined.groupby(["source", "label"]).size().unstack(fill_value=0))
    print("\nClass totals:")
    print(combined["label"].value_counts().reindex(TARGET_CLASSES, fill_value=0))
    print("\nGrand total:", len(combined))
    print("\nSaved manifest:", out)

if __name__ == "__main__":
    main()
