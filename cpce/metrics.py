"""Optional annotation-based evaluation; annotations do not enter clustering."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


def evaluate(y_true, y_pred, rare_fraction=0.05, smoothing=0.05, detection_threshold=0.05):
    """ARI and rare-class metrics from Eqs. (11)-(12).

    Match all annotated and predicted classes by maximum total F1.
    Unmatched annotated classes receive F1=0. With no rare classes,
    return None for the two rare-class summary metrics.
    """
    truth = np.asarray(y_true).astype(str)
    prediction = np.asarray(y_pred)
    if truth.ndim != 1 or prediction.ndim != 1 or len(truth) != len(prediction) or not len(truth):
        raise ValueError("Labels must be nonempty 1-D arrays of equal length.")
    if not 0 < rare_fraction <= 1 or not np.isfinite(smoothing) or smoothing <= 0 or not 0 <= detection_threshold <= 1:
        raise ValueError("Invalid rare-class metric parameters.")
    classes = np.unique(truth)
    table = contingency_matrix(truth, prediction)
    sizes = table.sum(axis=1)
    f1 = 2 * table / (sizes[:, None] + table.sum(axis=0)[None, :])
    rows, cols = linear_sum_assignment(-f1)
    scores = np.zeros(len(classes))
    scores[rows] = f1[rows, cols]
    rare = sizes / len(truth) <= rare_fraction
    rare_scores = scores[rare]
    return {
        "ari": float(adjusted_rand_score(truth, prediction)),
        "geo_rare_f1": float(max(0, np.exp(np.log(rare_scores + smoothing).mean()) - smoothing)) if rare.any() else None,
        "detection_rate": float((rare_scores > detection_threshold).mean()) if rare.any() else None,
        "rare_classes": {str(name): float(score) for name, score in zip(classes[rare], rare_scores)},
    }
