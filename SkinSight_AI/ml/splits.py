"""Deterministic source/class-stratified splits with linked identity groups."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def present(value):
    return pd.notna(value) and str(value).strip().lower() not in {'', 'nan', 'none'}


def linked_groups(frame):
    """Union patient/lesion/image identities and exact content duplicates transitively."""
    parents = list(range(len(frame)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, row in enumerate(frame.to_dict('records')):
        source = str(row['source'])
        keys = [('image', source, str(row['original_id']))]
        for column in ['patient_id', 'lesion_id']:
            value = row.get(column)
            if present(value):
                keys.append((column, source, str(value).strip()))
        if present(row.get('image_sha256')):
            keys.append(('sha256', str(row['image_sha256'])))
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    return pd.Series([f'group::{find(i)}' for i in range(len(frame))], index=frame.index)


def split_three_way(frame):
    """Approx. 60/20/20; fixed folds, no search using model scores or test outcomes."""
    frame = frame.copy().reset_index(drop=True)
    frame['_group'] = linked_groups(frame)
    strata = frame.source.astype(str) + '::' + frame.label.astype(str)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(splitter.split(frame, strata, groups=frame['_group']))
    validation_ids = folds[0][1]
    test_ids = folds[1][1]
    train_mask = ~frame.index.isin(list(validation_ids) + list(test_ids))
    train = frame.loc[train_mask].copy()
    validation = frame.iloc[validation_ids].copy()
    test = frame.iloc[test_ids].copy()
    parts = {'train': train, 'validation': validation, 'test': test}
    groups = [set(part['_group']) for part in parts.values()]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise RuntimeError('Linked identity group leakage detected')
    # Fail instead of silently evaluating a source/class missing from a split.
    required = set(strata)
    for name, part in parts.items():
        actual = set(part.source.astype(str) + '::' + part.label.astype(str))
        if required - actual:
            raise ValueError(f'{name} lacks source/class groups: {sorted(required-actual)}. '
                             'More independent groups or an explicitly designed split are needed.')
    return train, validation, test
