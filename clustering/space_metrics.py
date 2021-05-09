import numpy as np
import sklearn.metrics as metrics


def score_with_silhouette(X: np.array, labels: list):
    return metrics.silhouette_score(X, labels)


def score_with_calinski_harabasz(X: np.array, labels: list):
    return metrics.calinski_harabasz_score(X, labels)


def score_with_davies_bouldin(X: np.array, labels: list):
    return metrics.davies_bouldin_score(X, labels)
