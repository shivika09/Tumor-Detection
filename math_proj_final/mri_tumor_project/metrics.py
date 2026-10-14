"""
metrics.py  (Member 4)

Two jobs:
  A) Classification metrics: confusion matrix, accuracy, precision, recall, F1
     (written from scratch with NumPy, just like Member 2 wrote PCA from scratch).
  B) PCA reconstruction metrics: how much information is lost when we keep
     only k components (REUSES Member 2's project / reconstruct /
     reconstruction_error from pca_transform.py - nothing is re-implemented).

Label convention (from Member 1's data_loader.py):
    1 = tumor ("yes" folder)  -> this is the POSITIVE class
    0 = no tumor ("no" folder)
"""
import numpy as np

from pca_transform import project, reconstruct, reconstruction_error   # Member 2
from config import CLASS_NAMES


# ----------------------------------------------------------------------
# A) Classification metrics
# ----------------------------------------------------------------------
def _check_labels(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true has {y_true.size} values but y_pred has {y_pred.size}.")
    if y_true.size == 0:
        raise ValueError("Cannot compute metrics on an empty set.")
    unexpected = (set(np.unique(y_true)) | set(np.unique(y_pred))) - {0, 1}
    if unexpected:
        bad = sorted(int(u) for u in unexpected)
        raise ValueError(f"Unexpected labels {bad}; only 0 (no tumor) and 1 (tumor) are allowed.")
    return y_true, y_pred


def confusion_matrix(y_true, y_pred):
    """
    2x2 matrix, rows = TRUE class, columns = PREDICTED class:

                       predicted 0     predicted 1
        true 0 (no)       TN              FP
        true 1 (yes)      FN              TP

    (Same layout as scikit-learn's confusion_matrix.)
    """
    y_true, y_pred = _check_labels(y_true, y_pred)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def _safe_div(a, b):
    """a / b, but return 0.0 instead of crashing when b == 0."""
    return float(a / b) if b > 0 else 0.0


def metrics_from_confusion(cm):
    """All the numbers we report, computed from the 2x2 confusion matrix."""
    tn, fp = int(cm[0, 0]), int(cm[0, 1])
    fn, tp = int(cm[1, 0]), int(cm[1, 1])
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n": tn + fp + fn + tp,
        "accuracy": _safe_div(tp + tn, tn + fp + fn + tp),   # how often we are right overall
        "precision": precision,                              # of images we called "tumor", how many really were
        "recall": recall,                                    # of real tumors, how many we found (sensitivity)
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "specificity": _safe_div(tn, tn + fp),               # of healthy images, how many we cleared (extra)
    }


def evaluate(y_true, y_pred):
    """y_true, y_pred -> dict of metrics (+ the confusion matrix itself)."""
    cm = confusion_matrix(y_true, y_pred)
    out = metrics_from_confusion(cm)
    out["confusion_matrix"] = cm
    return out


def format_report(m, title="Test-set evaluation"):
    """Human-readable report with a plain-language interpretation."""
    lines = [
        title,
        "=" * len(title),
        f"Images evaluated : {m['n']}",
        "",
        "Confusion matrix (rows = true class, columns = predicted class)",
        f"                      predicted {CLASS_NAMES[0]:<9} predicted {CLASS_NAMES[1]}",
        f"  true {CLASS_NAMES[0]:<9}  {m['tn']:>10} (TN)          {m['fp']:>6} (FP)",
        f"  true {CLASS_NAMES[1]:<9}  {m['fn']:>10} (FN)          {m['tp']:>6} (TP)",
        "",
        f"Accuracy    : {m['accuracy']:.4f}   ({m['tp'] + m['tn']} of {m['n']} images correct)",
        f"Precision   : {m['precision']:.4f}   (when we say 'tumor', we are right this often)",
        f"Recall      : {m['recall']:.4f}   (share of real tumors we actually found)",
        f"F1-score    : {m['f1']:.4f}   (balance of precision and recall)",
        f"Specificity : {m['specificity']:.4f}   (share of healthy scans we correctly cleared)  [extra]",
        "",
        "How to read it",
        f"  TP = {m['tp']}: tumor images correctly flagged as tumor.",
        f"  TN = {m['tn']}: healthy images correctly cleared.",
        f"  FP = {m['fp']}: healthy images wrongly flagged as tumor (a false alarm - causes extra tests/worry).",
        f"  FN = {m['fn']}: tumor images the model MISSED (the most dangerous mistake in a medical setting).",
    ]
    if m["fn"] > m["fp"]:
        lines.append("  -> The model misses more tumors than it falsely alarms. For screening, recall matters most, so this is the number to improve.")
    elif m["fp"] > m["fn"]:
        lines.append("  -> The model raises more false alarms than it makes misses, which is the safer direction for screening, but precision suffers.")
    else:
        lines.append("  -> Misses and false alarms are balanced.")
    lines.append(f"  Note: with only {m['n']} test images, one image changes accuracy by about {100 / m['n']:.1f} percentage points, so treat the numbers as rough.")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# B) PCA reconstruction metrics (reusing Member 2's functions)
# ----------------------------------------------------------------------
def reconstruction_error_per_image(X, mean, components, k):
    """Mean squared pixel error for EACH image when only k components are kept. Shape (n_images,)."""
    X_hat = reconstruct(project(X, mean, components, k), mean, components)   # Member 2
    return np.mean((X - X_hat) ** 2, axis=1)


def reconstruction_error_curve(X, mean, components, ks):
    """Overall reconstruction MSE for several values of k (uses Member 2's reconstruction_error)."""
    return [reconstruction_error(X, mean, components, k) for k in ks]


def variance_kept_curve(X_train, X_train_pca, ks):
    """
    Fraction of the training variance kept by the first k components.

    Why this works: the training data projected on principal component j has
    variance exactly equal to eigenvalue j, and the sum of ALL eigenvalues is the
    total variance of the data. So no refitting is needed - we only use data
    that already exists.
    """
    total = np.var(X_train, axis=0, ddof=1).sum()
    return [float(np.var(X_train_pca[:, :k], axis=0, ddof=1).sum() / total) for k in ks]
