from typing import List, Set, Tuple

import numpy as np
import numba


Clusters = List[Set[int]] # assignments and centroids
Centroids = np.ndarray


def k_means(vectors: np.ndarray, num_clusters: int,
            max_iterations: int=100) -> Tuple[Clusters, Centroids]:
    minima, maxima = np.amin(vectors, axis=0), np.amax(vectors, axis=0)
    centroids = np.random.uniform(minima, maxima, (num_clusters, vectors.shape[1]))
    clusters = _init_clusters(centroids, vectors)

    still_changing, iteration = True, 0
    while still_changing and iteration < max_iterations:
        centroids = _compute_centroids(clusters, vectors)
        closest_ids = _closest_centers(centroids, vectors)
        count = _reassign_clusters(clusters, closest_ids)
        still_changing = count > 0
        iteration += 1

    return clusters, centroids


def find_cluster_outliers(vectors: np.ndarray, clusters: Clusters,
                          cutoff_threshold: float=3.0) -> List[int]:
    outliers = []
    for cluster in clusters:
        cluster_vecs = vectors[list(cluster)]
        outliers += find_outliers(cluster_vecs, cutoff_threshold)
    return outliers


# @numba.njit(fastmath=True)
def find_outliers(vectors: np.ndarray, cutoff_threshold: float=3.0) -> List[int]:
    num_vecs = vectors.shape[0]
    centroid = np.sum(vectors, axis=0) / num_vecs
    variance = np.sum((vectors - centroid)**2, axis=1)
    deviation = np.sum(np.sqrt(variance)) / num_vecs
    z_scores = variance / deviation
    outlier_ids = np.where(z_scores >= cutoff_threshold)[0]
    return outlier_ids.tolist()


@numba.njit(fastmath=True)
def _closest_centers(centers: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    num_features = vectors.shape[1]
    closest_ids = np.zeros((vectors.shape[0]), dtype=np.int64)

    for vec_id, vec in enumerate(vectors):
        closest_center_id = 0
        min_dist_sq = np.inf
        for center_id, center in enumerate(centers):
            dist_sq = 0
            for i in range(num_features):
                dist_sq += (vec[i] - center[i])**2
            if dist_sq < min_dist_sq:
                closest_center_id = center_id
                min_dist_sq = dist_sq
        closest_ids[vec_id] = closest_center_id

    return closest_ids


def _reassign_clusters(clusters: List[Set[int]], closest_ids: np.ndarray) -> int:
    ids_to_reassign = set()
    for vec_id, closest_cluster_id in enumerate(closest_ids):
        if vec_id not in clusters[closest_cluster_id]:
            ids_to_reassign.add(vec_id)

    for cluster in clusters:
        for id in ids_to_reassign:
            if id in cluster:
                cluster.remove(id)

    for vec_id in ids_to_reassign:
        new_cluster_id = closest_ids[vec_id]
        clusters[new_cluster_id].add(vec_id)

    return len(ids_to_reassign)


def _compute_centroids(clusters: List[Set[int]], vectors: np.ndarray) -> np.ndarray:
    centroids = np.zeros((len(clusters), vectors.shape[1]))
    for cluster_id, vec_ids in enumerate(clusters):
        if len(vec_ids) == 0: continue
        cluster_vecs = vectors[list(vec_ids)]
        centroids[cluster_id] = np.sum(cluster_vecs, axis=0) / cluster_vecs.shape[0]
    return centroids


def _init_clusters(centroids: np.ndarray, vectors: np.ndarray) -> List[Set[int]]:
    num_clusters = centroids.shape[0]
    closest_ids = _closest_centers(centroids, vectors)
    clusters = [set() for _ in range(num_clusters)]
    for vec_id, closest_cluster_id in enumerate(closest_ids):
        clusters[closest_cluster_id].add(vec_id)
    return clusters
