"""Fixed-label metrics including sensitivity, specificity, and calibration error."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score
from dataset import CLASS_NAMES


def summarize(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if not len(labels):
        raise ValueError('Cannot evaluate an empty dataset')
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(labels, predictions, labels=range(len(CLASS_NAMES)))
    total = int(matrix.sum())
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        tp = int(matrix[i, i]); fn = int(matrix[i].sum()) - tp
        fp = int(matrix[:, i].sum()) - tp; tn = total - tp - fn - fp
        binary = labels == i
        per_class[name] = {
            'support': tp + fn,
            'sensitivity': tp / (tp + fn) if tp + fn else None,
            'specificity': tn / (tn + fp) if tn + fp else None,
            'roc_auc': float(roc_auc_score(binary, probabilities[:, i])) if binary.any() and (~binary).any() else None,
        }
    confidence = probabilities.max(axis=1)
    correctness = predictions == labels
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= low) & ((confidence < high) if high < 1 else (confidence <= high))
        if mask.any():
            ece += float(mask.mean() * abs(correctness[mask].mean() - confidence[mask].mean()))
    return {
        'sample_count': len(labels),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'classification_report': classification_report(labels, predictions, labels=range(len(CLASS_NAMES)),
                                                      target_names=CLASS_NAMES, zero_division=0, output_dict=True),
        'confusion_matrix': matrix.tolist(), 'class_order': CLASS_NAMES,
        'per_class': per_class, 'expected_calibration_error_10_bins': ece,
        'calibration_note': 'Softmax probabilities are not clinically calibrated confidence estimates.',
    }
