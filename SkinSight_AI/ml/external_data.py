"""Audit a new evaluation/calibration manifest against all previously used records."""
import hashlib
from pathlib import Path
import pandas as pd
from dataset import CLASS_NAMES
from splits import linked_groups


def audit_external(manifest, reference_dir, additional_reference=None):
    frame = pd.read_csv(manifest).reset_index(drop=True)
    required = {'image_path', 'label', 'source', 'original_id', 'image_sha256'}
    if required - set(frame.columns) or frame.empty:
        raise ValueError('External manifest must be nonempty and include ' + ', '.join(sorted(required)))
    if frame[list(required)].isna().any().any() or not set(frame.label).issubset(CLASS_NAMES):
        raise ValueError('External manifest contains missing fields or unsupported labels')
    if frame.duplicated(['source', 'original_id']).any():
        raise ValueError('External manifest contains duplicate source/image identities')
    for row in frame.itertuples():
        with Path(row.image_path).open('rb') as file:
            actual = hashlib.file_digest(file, 'sha256').hexdigest()
        if actual != row.image_sha256:
            raise ValueError(f'Image checksum mismatch for {row.original_id}')
    references = []
    for name in ['train', 'validation', 'test']:
        path = Path(reference_dir) / f'{name}_split.csv'
        if not path.is_file():
            raise FileNotFoundError(f'All three reference partitions are required: {path}')
        references.append(pd.read_csv(path))
    if additional_reference is not None:
        references.append(pd.read_csv(additional_reference))
    known = pd.concat(references, ignore_index=True)
    combined = pd.concat([known, frame], ignore_index=True)
    groups = linked_groups(combined)
    if set(groups.iloc[:len(known)]) & set(groups.iloc[len(known):]):
        raise ValueError('External data overlaps previously used patient/lesion/image/hash groups')
    coverage = frame.groupby(['source', 'label']).size()
    audit = {'records': len(frame), 'known_group_overlap': False,
             'source_class_counts': {f'{source}::{label}': int(n) for (source, label), n in coverage.items()},
             'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
             'limitations': 'No overlap of recorded identities or exact-file hashes. Missing/aliased patient IDs and re-encoded near-duplicates can still hide overlap; source independence and ground-truth provenance require review.'}
    return frame, audit
