import numpy as np
import sklearn.metrics as metrics

from scipy.sparse import csr_matrix


def score_with_silhouette(X: np.array, labels: list):
    return metrics.silhouette_score(X, labels)


def score_with_calinski_harabasz(X: np.array, labels: list):
    return metrics.calinski_harabasz_score(X, labels)


def score_with_davies_bouldin(X: np.array, labels: list):
    return metrics.davies_bouldin_score(X, labels)


def score_with_Q_function(X: np.array, labels: list,
                          adjacency_matrix: csr_matrix,
                          transition_matrix: csr_matrix):
    # Create the assignment matrix, a matrix of size (n_nodes, n_clusters)
    # in which assignment_matrix[i, j] = 1 if and only if node i is in
    # cluster j and assignment_matrix[i, j] = 0 otherwise.
    n_nodes = X.shape[0]
    n_clusters = max(labels) + 1
    assignment_matrix = np.zeros((n_nodes, n_clusters))
    for i in range(n_nodes):
        assignment_matrix[i, labels[i]] = 1

    # Compute the weighted degree of each node.
    d = transition_matrix.diagonal()
    D = np.outer(d, d)

    # Compute the value of the Q-function
    vol_G = adjacency_matrix.sum()
    return csr_matrix.dot(assignment_matrix.T, vol_G * adjacency_matrix -
                          D).dot(assignment_matrix).diagonal().sum()
