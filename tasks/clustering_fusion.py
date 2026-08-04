# tasks/clustering_fusion.py
"""Co-association matrix fusion for dual-consensus clustering.

Given two independent label assignments for the same N tracks, produce a
single fused assignment via the co-association matrix approach:
each pair of tracks votes on whether they belong together, and the
consensus matrix is hierarchically clustered.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Threshold above which we fall back to voting-fusion (O(N)) instead of
# the full O(N^2) co-association matrix to keep memory bounded.
_COASSOC_MAX_TRACKS = 5000


def _fuse_coassociation(
    labels_a, labels_b,
    X_a, X_b,
    weight_a=0.5, weight_b=0.5,
    nmi_threshold=0.3,
):
    """Fuse two label assignments into one via co-association.

    Parameters
    ----------
    labels_a, labels_b : np.ndarray (N,)
        Cluster labels from two independent streams.
    X_a, X_b : np.ndarray (N, D_a) and (N, D_b)
        Original feature vectors (pre-PCA, scaled) for each stream.
    weight_a, weight_b : float
        Relative weights when building the fused feature space for scoring.
    nmi_threshold : float
        Not currently used at co-association time; reserved for future
        dynamic mode selection (e.g., fall back to voting when NMI is low).

    Returns
    -------
    fused_labels : np.ndarray (N,)
        Consensus labels from hierarchical clustering on the co-association
        matrix. Noise points (label -1) from either input are propagated.
    """
    n = len(labels_a)
    if n == 0:
        return np.array([], dtype=int)

    # Propagate noise labels — any track marked -1 by either stream stays -1
    noise_mask = (labels_a == -1) | (labels_b == -1)

    if n > _COASSOC_MAX_TRACKS:
        return _fuse_voting(labels_a, labels_b, X_a, X_b, weight_a, weight_b, noise_mask)

    # Build co-association matrix C (N x N), float32, range [0.0, 1.0]
    # C[i, j] = average of (agree_a + agree_b) where agree = 1 if same cluster
    agree_a = labels_a[:, None] == labels_a[None, :]   # N x N bool
    agree_b = labels_b[:, None] == labels_b[None, :]   # N x N bool
    C = (agree_a.astype(np.float32) + agree_b.astype(np.float32)) / 2.0

    # Enforce noise isolation — noise tracks agree with nothing
    if noise_mask.any():
        C[noise_mask, :] = 0.0
        C[:, noise_mask] = 0.0

    # Hierarchical clustering on dissimilarity (1 - C)
    from sklearn.cluster import AgglomerativeClustering

    # Determine cluster count: average of both streams' non-noise counts
    n_clusters_a = len(set(labels_a) - {-1})
    n_clusters_b = len(set(labels_b) - {-1})
    n_clusters = max(2, (n_clusters_a + n_clusters_b) // 2)
    n_clusters = min(n_clusters, n - 1)

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average',
    )
    fused_labels = model.fit_predict(1.0 - C)

    # Re-apply noise mask
    if noise_mask.any():
        fused_labels[noise_mask] = -1

    return fused_labels


def _fuse_voting(labels_a, labels_b, X_a, X_b, weight_a, weight_b, noise_mask):
    """O(N) voting fallback for large track sets.

    For each track, picks the cluster label from the stream with higher
    confidence (proximity to cluster center). Falls back to majority if
    confidence is equal.
    """
    n = len(labels_a)
    fused = np.full(n, -1, dtype=int)

    # Build per-stream centroids
    def _centroids(labels, X):
        centroids = {}
        for cid in set(labels):
            if cid == -1:
                continue
            mask = labels == cid
            centroids[cid] = X[mask].mean(axis=0)
        return centroids

    cents_a = _centroids(labels_a, X_a)
    cents_b = _centroids(labels_b, X_b)

    for i in range(n):
        if noise_mask[i]:
            continue
        la, lb = labels_a[i], labels_b[i]
        if la == -1:
            fused[i] = lb
        elif lb == -1:
            fused[i] = la
        elif la == lb:
            fused[i] = la
        else:
            # Pick the label whose centroid is closer to this point
            dist_a = np.linalg.norm(X_a[i] - cents_a[la]) if la in cents_a else np.inf
            dist_b = np.linalg.norm(X_b[i] - cents_b[lb]) if lb in cents_b else np.inf
            fused[i] = la if dist_a <= dist_b else lb

    return fused
