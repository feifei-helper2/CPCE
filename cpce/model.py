"""Density anchors, capacity-constrained matching, and sparse consensus."""

from collections import deque
import heapq
import logging
from numbers import Integral

import numpy as np
from scipy.cluster.hierarchy import cut_tree, linkage
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)
EPS = 1e-8


def _neighbor_distances(X, n_neighbors):
    k = min(n_neighbors, len(X) - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    distances, _ = nn.kneighbors(X)
    # Removing one zero distance excludes self, also when observations coincide.
    return distances[:, 1:].mean(axis=1), nn, k


def _density_peaks(X, density, block_size):
    """Evaluate Eq. (3) exactly, with strictly higher-density neighbors."""
    n = len(X)
    order = np.argsort(-density, kind="stable")
    sorted_density = density[order]
    delta = np.empty(n, dtype=float)
    for start in range(0, n, block_size):
        rows = order[start:start + block_size]
        boundary = np.searchsorted(-sorted_density, -density[rows], side="left")
        best = np.full(len(rows), np.inf)
        farthest = np.zeros(len(rows))
        limit = n if np.any(boundary == 0) else int(boundary.max())
        for offset in range(0, limit, block_size):
            cols = order[offset:offset + block_size]
            distances = cdist(X[rows], X[cols])
            farthest = np.maximum(farthest, distances.max(axis=1))
            valid = np.arange(offset, offset + len(cols))[None, :] < boundary[:, None]
            best = np.minimum(best, np.where(valid, distances, np.inf).min(axis=1))
        delta[rows] = np.where(boundary == 0, farthest, best)
    return density * delta


def _anchors(X, gamma, n_clusters):
    candidates = np.argsort(-gamma, kind="stable")[:min(len(X), 5 * n_clusters)]
    peaks = X[candidates]
    if n_clusters == 1:
        return peaks.mean(axis=0, keepdims=True)
    groups = cut_tree(linkage(peaks, method="average"), n_clusters=n_clusters).ravel()
    return np.vstack([peaks[groups == k].mean(axis=0) for k in range(n_clusters)])


def _capacities(n_cells, potential, alpha, buffer, floor):
    k = len(potential)
    ranked = np.argsort(-potential, kind="stable")
    profile = np.arange(1, k + 1, dtype=float) ** (-alpha)
    profile /= profile.sum()
    quotas = np.maximum(
        np.floor(n_cells * profile * (1 + buffer)),
        np.floor(n_cells / k * floor),
    ).astype(np.int64)
    capacities = np.empty(k, dtype=np.int64)
    capacities[ranked] = quotas
    if capacities.sum() < n_cells:
        raise ValueError("Integer capacities do not cover all cells; increase capacity_buffer or capacity_floor.")
    return capacities


def _deferred_acceptance(distances, density, capacities):
    """Cell-proposing Gale-Shapley matching; ties favor lower cell/anchor IDs."""
    n, k = distances.shape
    if capacities.shape != (k,) or np.any(capacities < 0) or capacities.sum() < n:
        raise ValueError("Capacities must be nonnegative and sum to at least the number of cells.")
    preferences = np.argsort(distances, axis=1, kind="stable")
    utility = density[:, None] / (distances ** 2 + EPS)
    next_choice = np.zeros(n, dtype=int)
    free = deque(range(n))
    accepted = [[] for _ in range(k)]
    while free:
        cell = free.popleft()
        if next_choice[cell] == k:
            raise RuntimeError("Preference list exhausted despite sufficient total capacity.")
        anchor = int(preferences[cell, next_choice[cell]])
        next_choice[cell] += 1
        # The heap root is the least preferred accepted cell.
        heapq.heappush(accepted[anchor], (utility[cell, anchor], -cell))
        if len(accepted[anchor]) > capacities[anchor]:
            _, rejected = heapq.heappop(accepted[anchor])
            free.append(-rejected)
    labels = np.full(n, -1, dtype=int)
    for anchor, cells in enumerate(accepted):
        for _, cell in cells:
            labels[-cell] = anchor
    return labels


def _incidence(partitions, n_clusters):
    n, rounds = partitions.shape
    rows = np.repeat(np.arange(n), rounds)
    cols = (partitions + n_clusters * np.arange(rounds)).ravel()
    return csr_matrix((np.ones(n * rounds, dtype=np.float64), (rows, cols)),
                      shape=(n, rounds * n_clusters))


def _graph_consensus(incidence, n_clusters, resolution, random_state):
    import igraph as ig
    import leidenalg

    n, m = incidence.shape
    memberships = incidence.tocoo()
    edges = np.column_stack((memberships.row, n + memberships.col))
    graph = ig.Graph(n=n + m, edges=edges.tolist(), directed=False)
    graph.vs["type"] = [False] * n + [True] * m
    # Each edge is one membership; no cell-cell projection is materialized.
    for _ in range(32):
        partition = leidenalg.find_partition(
            graph, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution, seed=int(random_state), n_iterations=-1,
        )
        _, labels = np.unique(np.asarray(partition.membership[:n]), return_inverse=True)
        if len(np.unique(labels)) >= n_clusters:
            return labels, resolution
        resolution *= 2
    raise RuntimeError("Leiden did not produce enough cell groups; increase consensus_resolution.")


def _complete_linkage(X, labels, n_clusters, block_size):
    """Merge fragments using maximum cell-to-cell distances (Eq. 10)."""
    _, labels = np.unique(labels, return_inverse=True)
    groups = [np.flatnonzero(labels == k) for k in range(labels.max() + 1)]
    if len(groups) == n_clusters:
        return labels
    if len(groups) < n_clusters:
        raise ValueError("Consolidation requires at least n_clusters fragments.")
    distances = np.empty(len(groups) * (len(groups) - 1) // 2, dtype=float)
    position = 0
    for a, left in enumerate(groups):
        for right in groups[a + 1:]:
            maximum = 0.0
            for i in range(0, len(left), block_size):
                for j in range(0, len(right), block_size):
                    maximum = max(maximum, float(cdist(
                        X[left[i:i + block_size]], X[right[j:j + block_size]]
                    ).max()))
            distances[position] = maximum
            position += 1
    merged = cut_tree(linkage(distances, method="complete"), n_clusters=n_clusters).ravel()
    return merged[labels]


class CPCE:
    """Fixed-K clustering of a finite, dense cell-by-feature representation.

    Parameters follow the paper's density, capacity, and ensemble stages.
    ``alpha_max`` is the largest exponent in a linear schedule from 1;
    with one round, that round uses ``alpha_max``. Density perturbations
    have standard deviation ``density_noise`` times each mean kNN distance.
    ``capacity_floor`` is omega in Eq. (6), an upper-capacity floor rather
    than a minimum cluster size. ``block_size`` bounds distance workspaces.
    """

    def __init__(self, n_clusters, n_rounds=10, alpha_max=2.0, n_neighbors=15,
                 capacity_buffer=0.15, capacity_floor=1.0, density_noise=0.01,
                 consensus_resolution=2.0, random_state=42, block_size=512):
        self.n_clusters = n_clusters
        self.n_rounds = n_rounds
        self.alpha_max = alpha_max
        self.n_neighbors = n_neighbors
        self.capacity_buffer = capacity_buffer
        self.capacity_floor = capacity_floor
        self.density_noise = density_noise
        self.consensus_resolution = consensus_resolution
        self.random_state = random_state
        self.block_size = block_size

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty, finite 2-D cell-by-feature array.")
        for name in ("n_clusters", "n_rounds", "n_neighbors", "block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed the number of cells.")
        for name, minimum, strict in (("alpha_max", 1, False),
                                      ("capacity_buffer", 0, True),
                                      ("capacity_floor", 1, False),
                                      ("density_noise", 0, False),
                                      ("consensus_resolution", 0, True)):
            value = getattr(self, name)
            if not np.isfinite(value) or value < minimum or (strict and value == minimum):
                raise ValueError(f"Invalid {name}: {value}.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral) or not 0 <= self.random_state <= 2**31 - 1:
            raise ValueError("random_state must be an integer in [0, 2**31 - 1].")

        n, k = len(X), self.n_clusters
        self.n_features_in_ = X.shape[1]
        if k == 1 or k == n:
            self.labels_ = np.zeros(n, dtype=int) if k == 1 else np.arange(n)
            self.base_labels_ = np.tile(self.labels_[:, None], (1, self.n_rounds))
            self.capacities_ = np.tile(np.bincount(self.labels_, minlength=k), (self.n_rounds, 1))
            self.incidence_ = _incidence(self.base_labels_, k)
            self.anchors_ = np.vstack([X[self.labels_ == j].mean(axis=0) for j in range(k)])
            self.density_ = np.ones(n)
            self.consensus_labels_ = self.labels_.copy()
            self.resolution_ = self.consensus_resolution
            self.alphas_ = np.linspace(1, self.alpha_max, self.n_rounds) if self.n_rounds > 1 else np.array([self.alpha_max])
            return self

        logger.info("Selecting density peak anchors (%d cells, %d clusters)", n, k)
        mean_distance, neighbors, knn = _neighbor_distances(X, self.n_neighbors)
        self.density_ = 1 / (mean_distance + EPS)
        gamma = _density_peaks(X, self.density_, self.block_size)
        self.anchors_ = _anchors(X, gamma, k)
        core_indices = neighbors.kneighbors(self.anchors_, n_neighbors=knn, return_distance=False)
        distances = cdist(X, self.anchors_)
        self.alphas_ = np.linspace(1, self.alpha_max, self.n_rounds) if self.n_rounds > 1 else np.array([self.alpha_max])
        self.base_labels_ = np.empty((n, self.n_rounds), dtype=int)
        self.capacities_ = np.empty((self.n_rounds, k), dtype=np.int64)
        rng = np.random.default_rng(self.random_state)
        for t, alpha in enumerate(self.alphas_):
            jitter = rng.normal(0, self.density_noise, n)
            perturbed_distance = np.maximum(mean_distance * (1 + jitter), 0)
            density = 1 / (perturbed_distance + EPS)
            potential = density[core_indices].sum(axis=1)
            capacities = _capacities(n, potential, alpha, self.capacity_buffer, self.capacity_floor)
            self.capacities_[t] = capacities
            self.base_labels_[:, t] = _deferred_acceptance(distances, density, capacities)
            logger.info("Base partition %d/%d", t + 1, self.n_rounds)

        self.incidence_ = _incidence(self.base_labels_, k)
        self.consensus_labels_, self.resolution_ = _graph_consensus(
            self.incidence_, k, self.consensus_resolution, self.random_state)
        logger.info("Consolidating %d consensus groups to %d", len(np.unique(self.consensus_labels_)), k)
        self.labels_ = _complete_linkage(X, self.consensus_labels_, k, self.block_size)
        return self

    def fit_predict(self, X):
        """Return one integer label per cell, with exactly n_clusters groups."""
        return self.fit(X).labels_
