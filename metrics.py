"""Regression and zero-inclusive/zero-exclusive sentiment metrics."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def regression_metrics(prediction, target):
    prediction, target = np.asarray(prediction).reshape(-1), np.asarray(target).reshape(-1)
    nonzero = target != 0
    return {
        'Has0_acc_2': float(accuracy_score(target >= 0, prediction >= 0)),
        'Has0_F1': float(f1_score(target >= 0, prediction >= 0, average='weighted')),
        'Non0_acc_2': float(accuracy_score(target[nonzero] > 0, prediction[nonzero] > 0)),
        'Non0_F1': float(f1_score(target[nonzero] > 0, prediction[nonzero] > 0, average='weighted')),
        'Acc_7': float(np.mean(np.round(np.clip(prediction, -3, 3)) == np.round(np.clip(target, -3, 3)))),
        'MAE': float(np.mean(np.abs(prediction - target))),
        'Corr': float(np.corrcoef(prediction, target)[0, 1]),
    }
