# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License v3.0. See the LICENSE file
# in the project root or <https://github.com/NeptuneHub/AudioMuse-AI/blob/main/LICENSE>

"""Per-iteration clustering worker: parameter generation, fitting and scoring.
The inner loop of the clustering search run by tasks.clustering. Given a method
and parameter set it prepares and scales the feature/embedding data, fits a model
(via clustering_gpu), and scores the resulting playlists. Also generates the
random and evolutionary parameter mutations that the elitist search explores.
Main Features:
* _perform_single_clustering_iteration / _apply_clustering_model: run one
  clustering attempt end to end and return a scored result.
* _split_oversized_clusters: DBSCAN components larger than
  CLUSTERING_MAX_PLAYLIST_SONGS are re-split with KMeans into playlist-sized
  chunks - music embeddings form one connected density mass, so raw DBSCAN
  either merges everything into a single giant cluster or marks it all noise.
* _generate_random_parameters / _mutate_parameters / _generate_evolutionary_parameters:
  sample and mutate KMeans/DBSCAN/GMM/spectral/PCA params within configured ranges.
* Playlist shaping helpers (chunking, shuffling, optional AI naming) for each run.
"""

import json
import random
import logging
import time
import numpy as np
from collections import defaultdict
# time, re, and cdist imports moved to clustering_postprocessing.py

# Sklearn imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

logger = logging.getLogger(__name__)

# GPU clustering support (optional)
try:
    from .clustering_gpu import get_clustering_model, get_pca_model
    GPU_CLUSTERING_AVAILABLE = True
except ImportError:
    GPU_CLUSTERING_AVAILABLE = False
    logger.debug("GPU clustering module not available, using CPU only")

# RQ imports for safe result fetching
from rq.job import Job, JobStatus
from rq.exceptions import NoSuchJobError

from config import (STRATIFIED_GENRES, OTHER_FEATURE_LABELS, MOOD_LABELS, MAEST_MOOD_LABELS,
                    MAX_DISTANCE, MAX_SONGS_PER_ARTIST, GMM_COVARIANCE_TYPE, SPECTRAL_N_NEIGHBORS,
                    TOP_K_MOODS_FOR_PURITY_CALCULATION, LN_MOOD_DIVERSITY_STATS,
                    LN_MOOD_PURITY_STATS, LN_MOOD_DIVERSITY_EMBEDING_STATS,
                    LN_MOOD_PURITY_EMBEDING_STATS, LN_OTHER_FEATURES_DIVERSITY_STATS,
                    LN_OTHER_FEATURES_PURITY_STATS,
                    LN_MAEST_GENRE_DIVERSITY_STATS, LN_MAEST_GENRE_PURITY_STATS,
                    LN_HYBRID_MOOD_DIVERSITY_STATS, LN_HYBRID_MOOD_PURITY_STATS,
                    OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY,
                    USE_GPU_CLUSTERING, TASK_STATUS_SUCCESS,CLUSTERING_SUBSET_SONGS,
                    HYBRID_PCA_MUSICNN, HYBRID_PCA_MAEST,
                    HYBRID_WEIGHT_MUSICNN, HYBRID_WEIGHT_MAEST)
from .commons import score_vector

# Low-level DB / queue primitives, imported directly rather than via the
# app_helper facade (keeps this helper decoupled from the blueprint layer).
from database import get_tracks_by_ids, get_score_data_by_ids, get_task_info_from_db
from taskqueue import redis_conn


# --- Playlist Naming & Shuffling Helpers ---

def _shuffle_playlist_songs(songs, playlist_name):
    """Fisher-Yates shuffle a list of song tuples; log the result."""
    final_songs = songs.copy()
    n = len(final_songs)
    if n <= 1:
        logger.info("FINAL: '%s' has only %d songs - no shuffling needed", playlist_name, n)
        return final_songs

    current_time_seed = int(time.time() * 1000000) % 1000000
    for i in range(n - 1, 0, -1):
        j = (random.randint(0, i) + current_time_seed + i * 7) % (i + 1)
        final_songs[i], final_songs[j] = final_songs[j], final_songs[i]
        current_time_seed = (current_time_seed * 1103515245 + 12345) % (2 ** 31)

    logger.info("FINAL FISHER-YATES SHUFFLE applied to '%s': %d songs", playlist_name, len(final_songs))
    logger.info("FINAL ORDER: First song = '%s', Last song = '%s'", final_songs[0][1], final_songs[-1][1])
    return final_songs


def _assign_playlist_chunks(final_songs, max_songs, base_name, final_playlists):
    """Chunk oversized playlists or store the list as-is."""
    if max_songs > 0 and len(final_songs) > max_songs:
        chunks = [final_songs[i:i + max_songs] for i in range(0, len(final_songs), max_songs)]
        for idx, chunk in enumerate(chunks, 1):
            final_playlists[f"{base_name} ({idx})"] = chunk
    else:
        final_playlists[base_name] = final_songs


def _try_ai_name_playlist(original_name, songs, centroids, ai_provider,
                          ollama_url, ollama_model, openai_url, openai_model, openai_key,
                          gemini_key, gemini_model, mistral_key, mistral_model):
    """Attempt AI naming; return the original name on failure."""
    # Import AI naming for playlist helpers
    from .ai.api import get_ai_playlist_name
    from .ai.prompts import creative_prompt_template

    ai_config = {
        'provider': ai_provider,
        'ollama_url': ollama_url, 'ollama_model': ollama_model,
        'openai_url': openai_url, 'openai_model': openai_model, 'openai_key': openai_key,
        'gemini_key': gemini_key, 'gemini_model': gemini_model,
        'mistral_key': mistral_key, 'mistral_model': mistral_model,
    }
    ai_name = get_ai_playlist_name(
        creative_prompt_template,
        [{'title': s_title, 'author': s_author} for _, s_title, s_author in songs],
        centroids.get(original_name, {}),
        ai_config,
    )
    if ai_name and "Error" not in ai_name:
        return ai_name.strip().replace("\n", " ")
    logger.warning("AI naming failed for '%s': %s. Using original name.", original_name, ai_name)
    return original_name


# --- Main Orchestrator for a Single Iteration ---

def _perform_single_clustering_iteration(
    run_idx, item_ids_for_subset,
    clustering_method, num_clusters_min_max, dbscan_params_ranges, gmm_params_ranges,
    spectral_params_ranges, pca_params_ranges, active_mood_labels,
    max_songs_per_cluster, log_prefix,
    elite_solutions_params_list, exploitation_probability, mutation_config,
    score_weights, enable_clustering_embeddings, clustering_mode="musicnn",
    hybrid_pca_musicnn=HYBRID_PCA_MUSICNN, hybrid_pca_maest=HYBRID_PCA_MAEST,
    nmi_threshold=0.3):
    """
    Orchestrates a single evolutionary run of the clustering process.
    This function is now a high-level coordinator.
    """
    try:
        # Local import to prevent circular dependency
        from flask_app import app

        if not item_ids_for_subset:
            logger.warning(f"{log_prefix} Iteration {run_idx}: Received empty item ID subset. Skipping.")
            return {"fitness_score": -1.0}

        # 1. Prepare Data: Fetch full track data and create feature vectors based on clustering_mode
        valid_tracks = None
        X_feat_musicnn = None
        X_feat_maest = None
        X_feat_orig = None
        X_embed_raw = None
        data_to_cluster = None
        scaler = None
        use_hybrid_internal_pca = False

        with app.app_context():
            # Determine which mood labels to use based on clustering mode
            if clustering_mode == "maest" or clustering_mode == "hybrid_blend" or clustering_mode == "dual_consensus":
                # All dual-data modes share the same data fetch
                valid_tracks, X_feat_musicnn, X_feat_maest = _prepare_iteration_data_both(
                    item_ids_for_subset, MOOD_LABELS, MAEST_MOOD_LABELS, log_prefix, run_idx
                )
                if valid_tracks is None:
                    return {"fitness_score": -1.0}

                # Build data_to_cluster based on mode
                if clustering_mode == "maest":
                    if X_feat_maest is None or X_feat_maest.shape[0] == 0:
                        logger.error(f"{log_prefix} Iteration {run_idx}: No MAEST features available.")
                        return {"fitness_score": -1.0}
                    scaler = StandardScaler()
                    data_to_cluster = scaler.fit_transform(X_feat_maest)
                    X_feat_orig = X_feat_maest
                    active_mood_labels = MAEST_MOOD_LABELS

                elif clustering_mode == "hybrid_blend":
                    if X_feat_musicnn is None or X_feat_musicnn.shape[0] == 0:
                        logger.error(f"{log_prefix} Iteration {run_idx}: No MusicNN features available for hybrid blend.")
                        return {"fitness_score": -1.0}
                    if X_feat_maest is None or X_feat_maest.shape[0] == 0:
                        logger.error(f"{log_prefix} Iteration {run_idx}: No MAEST features available for hybrid blend.")
                        return {"fitness_score": -1.0}
                    data_to_cluster = (X_feat_musicnn, X_feat_maest)  # Tuple marker
                    use_hybrid_internal_pca = True
                    X_feat_orig = X_feat_musicnn
                    active_mood_labels = MOOD_LABELS

                else:  # dual_consensus
                    if X_feat_musicnn is None or X_feat_musicnn.shape[0] == 0:
                        logger.error(f"{log_prefix} Iteration {run_idx}: No MusicNN features for dual consensus.")
                        return {"fitness_score": -1.0}
                    if X_feat_maest is None or X_feat_maest.shape[0] == 0:
                        logger.error(f"{log_prefix} Iteration {run_idx}: No MAEST features for dual consensus.")
                        return {"fitness_score": -1.0}
                    data_to_cluster = (X_feat_musicnn, X_feat_maest)  # Tuple marker
                    use_hybrid_internal_pca = False
                    X_feat_orig = X_feat_musicnn
                    active_mood_labels = MOOD_LABELS

            else:
                # Default musicnn mode: use existing logic
                if enable_clustering_embeddings:
                    valid_tracks, X_feat_orig, X_embed_raw = _prepare_iteration_data(
                        item_ids_for_subset, active_mood_labels, True, log_prefix, run_idx
                    )
                    if valid_tracks is None:
                        return {"fitness_score": -1.0}
                    data_to_cluster, scaler = _prepare_and_scale_data(X_feat_orig, X_embed_raw, True)
                else:
                    valid_tracks, X_feat_orig, X_embed_raw = _prepare_iteration_data(
                        item_ids_for_subset, active_mood_labels, False, log_prefix, run_idx
                    )
                    if valid_tracks is None:
                        return {"fitness_score": -1.0}
                    data_to_cluster, scaler = _prepare_and_scale_data(X_feat_orig, X_embed_raw, False)

        if data_to_cluster is None:
            logger.error(f"{log_prefix} Iteration {run_idx}: Data for clustering is empty after prep. Cannot proceed.")
            return {"fitness_score": -1.0}

        # 2b. Handle hybrid blend PCA BEFORE parameter generation (uses fixed config values)
        # This is necessary because _generate_evolutionary_parameters needs data.shape attributes
        hybrid_pca_models = None
        hybrid_pca_reduced = None  # (X_musicnn_pca, X_maest_pca) — stored for evolved-weight re-concat
        if use_hybrid_internal_pca and isinstance(data_to_cluster, tuple):
            X_musicnn_raw, X_maest_raw = data_to_cluster

            # Use fixed HYBRID_PCA_* values (evolutionary PCA is skipped for hybrid mode)
            m_scaler, m_pca, X_musicnn_pca = _pca_reduce(X_musicnn_raw, hybrid_pca_musicnn, log_prefix, "MusicNN")
            a_scaler, a_pca, X_maest_pca = _pca_reduce(X_maest_raw, hybrid_pca_maest, log_prefix, "MAEST")

            if X_musicnn_pca is None or X_maest_pca is None:
                logger.error(f"{log_prefix} Iteration {run_idx}: PCA reduction failed for hybrid blend.")
                return {"fitness_score": -1.0}

            # Store PCA-reduced streams so we can re-concatenate with evolved weights
            hybrid_pca_reduced = (X_musicnn_pca, X_maest_pca)

            # Temporarily concatenate with default weights so _generate_evolutionary_parameters
            # can read data.shape (shape is weight-independent).
            data_to_cluster = np.hstack([X_musicnn_pca, X_maest_pca])

            # Store PCA models for later inversion (HybridScaler in Task 2b.6)
            hybrid_pca_models = {
                'musicnn': {'scaler': m_scaler, 'pca': m_pca, 'output_dims': X_musicnn_pca.shape[1]},
                'maest': {'scaler': a_scaler, 'pca': a_pca, 'output_dims': X_maest_pca.shape[1]}
            }

        # 3. Generate Parameters BEFORE the dual_consensus early-return block,
        #    because that block needs params (PCA configs, clustering method) to run.
        params = _generate_evolutionary_parameters(
            elite_solutions_params_list, exploitation_probability, mutation_config,
            clustering_method, data_to_cluster, pca_params_ranges,
            num_clusters_min_max, dbscan_params_ranges, gmm_params_ranges, spectral_params_ranges,
            log_prefix, run_idx, clustering_mode=clustering_mode
        )

        # Attach hybrid PCA models to params and disable regular PCA for hybrid mode
        if hybrid_pca_models:
            params['_hybrid_pca_models'] = hybrid_pca_models
            # Read evolved fusion weights from params (fall back to config constants)
            weights = params.get('hybrid_weights', {'musicnn': HYBRID_WEIGHT_MUSICNN, 'maest': HYBRID_WEIGHT_MAEST})
            w_mn = weights['musicnn']
            w_ma = weights['maest']
            params['_hybrid_weights'] = {'musicnn': w_mn, 'maest': w_ma}
            params['pca_config']['enabled'] = False

            # Re-concatenate with the evolved weights (data_to_cluster currently holds
            # the unweighted concat from the temporary step above).
            X_musicnn_pca, X_maest_pca = hybrid_pca_reduced
            X_musicnn_weighted = X_musicnn_pca * w_mn
            X_maest_weighted = X_maest_pca * w_ma
            data_to_cluster = np.hstack([X_musicnn_weighted, X_maest_weighted])

        # ─── Dual Consensus: run both streams independently, fuse ──────────────
        if clustering_mode == "dual_consensus" and isinstance(data_to_cluster, tuple):
            X_musicnn_raw, X_maest_raw = data_to_cluster
            from .clustering_fusion import _fuse_coassociation

            # Params are now generated at shared step 3 with clustering_mode='dual_consensus',
            # yielding per-stream PCA configs (params['musicnn_pca'], params['maest_pca'])

            pca_mn, pca_ma = None, None
            scaler_mn = StandardScaler()
            scaled_mn = scaler_mn.fit_transform(X_musicnn_raw)
            mn_pca_cfg = params.get('musicnn_pca', params['pca_config'])
            if mn_pca_cfg['enabled'] and mn_pca_cfg['components'] < X_musicnn_raw.shape[1]:
                pca_mn = PCA(n_components=mn_pca_cfg['components'])
                reduced_mn = pca_mn.fit_transform(scaled_mn)
            else:
                reduced_mn = scaled_mn

            # Stream 2: MAEST — uses its own PCA config from params
            scaler_ma = StandardScaler()
            scaled_ma = scaler_ma.fit_transform(X_maest_raw)
            ma_pca_cfg = params.get('maest_pca', params['pca_config'])
            if ma_pca_cfg['enabled'] and ma_pca_cfg['components'] < X_maest_raw.shape[1]:
                pca_ma = PCA(n_components=min(ma_pca_cfg['components'], X_maest_raw.shape[1], X_maest_raw.shape[0] - 1))
                reduced_ma = pca_ma.fit_transform(scaled_ma)
            else:
                reduced_ma = scaled_ma

            # Cluster both streams independently
            labels_mn, centers_mn, _ = _apply_clustering_model(
                reduced_mn, params['clustering_method_config'], log_prefix, run_idx
            )
            labels_ma, centers_ma, _ = _apply_clustering_model(
                reduced_ma, params['clustering_method_config'], log_prefix, run_idx
            )

            if labels_mn is None or labels_ma is None:
                logger.error(f"{log_prefix} Iteration {run_idx}: Dual consensus clustering failed on one or both streams.")
                return {"fitness_score": -1.0}

            # Read evolved fusion weights from params (fall back to config constants)
            weights = params.get('hybrid_weights', {'musicnn': HYBRID_WEIGHT_MUSICNN, 'maest': HYBRID_WEIGHT_MAEST})
            weight_mn = weights['musicnn']
            weight_ma = weights['maest']

            # Fuse labels via co-association
            fused_labels = _fuse_coassociation(
                labels_mn, labels_ma,
                X_a=reduced_mn, X_b=reduced_ma,
                weight_a=weight_mn, weight_b=weight_ma,
                nmi_threshold=nmi_threshold,
            )

            # Build fused feature space for scoring (weighted concat of each stream in original space)
            # Reuse the per-stream scalers/PCAs from above for the combined space
            m_pca_reduced = reduced_mn * weight_mn
            a_pca_reduced = reduced_ma * weight_ma
            data_for_metrics = np.hstack([m_pca_reduced, a_pca_reduced])

            # Build fused centroid map
            fused_centers = {}
            for cid in set(fused_labels):
                if cid == -1:
                    continue
                mask = fused_labels == cid
                if mask.sum() > 0:
                    fused_centers[cid] = data_for_metrics[mask].mean(axis=0)

            # Build HybridScaler metadata dict for centroid inversion during naming
            params['_hybrid_pca_models'] = {
                'musicnn': {'scaler': scaler_mn, 'pca': pca_mn, 'output_dims': reduced_mn.shape[1]},
                'maest': {'scaler': scaler_ma, 'pca': pca_ma, 'output_dims': reduced_ma.shape[1]},
            }
            params['_hybrid_weights'] = {'musicnn': weight_mn, 'maest': weight_ma}
            params['pca_config']['enabled'] = False

            return _format_and_score_iteration_result(
                fused_labels, valid_tracks, X_feat_orig, data_for_metrics,
                fused_centers, None, None, None, active_mood_labels,
                params, max_songs_per_cluster, run_idx, enable_clustering_embeddings, score_weights, log_prefix,
                clustering_mode=clustering_mode,
            )

        # 4. Apply PCA if specified by the generated parameters (skipped for hybrid_blend)
        pca_model, data_after_pca = None, data_to_cluster
        if params['pca_config']['enabled']:
            # Use GPU PCA if available and enabled
            if USE_GPU_CLUSTERING and GPU_CLUSTERING_AVAILABLE:
                pca_model = get_pca_model(n_components=params['pca_config']['components'], use_gpu=True)
            else:
                pca_model = PCA(n_components=params['pca_config']['components'])

            data_after_pca = pca_model.fit_transform(data_to_cluster)
            params['pca_config']['components'] = pca_model.n_components_ # Update with actual components

        # 5. Apply the chosen clustering model
        labels, cluster_centers_map, model = _apply_clustering_model(
            data_after_pca, params['clustering_method_config'], log_prefix, run_idx
        )
        if labels is None: # Clustering failed
            return {"fitness_score": -1.0}

        # 6. Format results and calculate fitness score
        return _format_and_score_iteration_result(
            labels, valid_tracks, X_feat_orig, data_after_pca,
            cluster_centers_map, model, pca_model, scaler, active_mood_labels,
            params, max_songs_per_cluster, run_idx, enable_clustering_embeddings, score_weights, log_prefix,
            clustering_mode=clustering_mode
        )

    except Exception:
        logger.error(f"{log_prefix} Iteration {run_idx} failed critically", exc_info=True)
        raise

# --- Step 1: Data Preparation ---

def _prepare_iteration_data(item_ids, active_mood_labels, use_embeddings, log_prefix, run_idx):
    """Fetches track data, creates feature/embedding vectors, and ensures alignment."""

    logger.info(f"{log_prefix} Iteration {run_idx}: Fetching data for {len(item_ids)} tracks. Use embeddings: {use_embeddings}")
    rows = get_tracks_by_ids(item_ids) if use_embeddings else get_score_data_by_ids(item_ids) # These functions are now imported locally
    valid_tracks, X_feat_orig_list, X_embed_raw_list = [], [], []
    for row_data in (dict(r) for r in rows if r):
        try:
            feature_vec = score_vector(row_data, active_mood_labels, OTHER_FEATURE_LABELS)
            if use_embeddings:
                embedding_vec = row_data.get('embedding_vector')
                if embedding_vec is None or embedding_vec.size == 0:
                    logger.warning(f"Skipping track {row_data.get('item_id')} due to missing embedding.")
                    continue
                X_embed_raw_list.append(embedding_vec)
            X_feat_orig_list.append(feature_vec)
            valid_tracks.append(row_data)
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Skipping track {row_data.get('item_id')} due to data parsing error.")
    if not valid_tracks:
        logger.error(f"{log_prefix} Iteration {run_idx}: No valid tracks could be processed.")
        return None, None, None
    return valid_tracks, np.array(X_feat_orig_list), np.array(X_embed_raw_list) if use_embeddings else None

def _prepare_iteration_data_both(item_ids, musicnn_labels, maest_labels, log_prefix, run_idx):
    """Returns (valid_tracks, X_feat_musicnn, X_feat_maest).

    Both feature vectors from the same track subset.
    Either can be None if that model's data column is empty for all tracks.
    """
    rows = get_score_data_by_ids(item_ids)  # Now includes maest_mood_vector (Task 3)
    valid_tracks, X_mn_list, X_ma_list = [], [], []
    for row_data in (dict(r) for r in rows if r):
        try:
            feat_mn = score_vector(row_data, musicnn_labels, OTHER_FEATURE_LABELS, mood_column='mood_vector')
            feat_ma = score_vector(row_data, maest_labels, OTHER_FEATURE_LABELS, mood_column='maest_mood_vector')
            valid_tracks.append(row_data)
            X_mn_list.append(feat_mn)
            X_ma_list.append(feat_ma)
        except (json.JSONDecodeError, TypeError):
            continue
    X_mn = np.array(X_mn_list) if X_mn_list else None
    X_ma = np.array(X_ma_list) if X_ma_list else None
    return valid_tracks, X_mn, X_ma

class HybridScaler:
    """Handles inversion of hybrid-space vectors back to original musicnn feature space.

    The hybrid blend process is:
    1. Scale musicnn features -> PCA-reduce -> weight-scale
    2. Scale maest features -> PCA-reduce -> weight-scale
    3. Concatenate weighted vectors

    To invert for naming, we need to:
    1. Split hybrid vector into musicnn and maest portions
    2. Un-weight each portion (divide by weights)
    3. Inverse PCA for each
    4. Inverse scale for each
    5. Return only the musicnn portion (used for naming with mood labels)
    """
    def __init__(self, m_scaler, m_pca, a_scaler, a_pca, n_musicnn_dims,
                 weight_musicnn=HYBRID_WEIGHT_MUSICNN, weight_maest=HYBRID_WEIGHT_MAEST):
        """Initialize with the PCA models and scalers from hybrid blending.

        Args:
            m_scaler: StandardScaler for musicnn features
            m_pca: PCA model for musicnn features (can be None if no PCA applied)
            a_scaler: StandardScaler for maest features
            a_pca: PCA model for maest features (can be None if no PCA applied)
            n_musicnn_dims: Original dimensionality of musicnn feature space
            weight_musicnn: Evolved fusion weight for the musicnn stream (defaults
                to the config constant for backward compatibility)
            weight_maest: Evolved fusion weight for the maest stream (defaults
                to the config constant for backward compatibility)
        """
        self.m_scaler = m_scaler
        self.m_pca = m_pca
        self.a_scaler = a_scaler
        self.a_pca = a_pca
        self.n_musicnn_dims = n_musicnn_dims
        self.weight_musicnn = weight_musicnn
        self.weight_maest = weight_maest

    def inverse_transform(self, hybrid_vec, musicnn_output_dims=None, maest_output_dims=None):
        """Invert a hybrid-space vector back to original musicnn feature space.

        Args:
            hybrid_vec: 1D numpy array in hybrid blended space
            musicnn_output_dims: PCA output dims for musicnn (from stored config)
            maest_output_dims: PCA output dims for maest (from stored config)

        Returns:
            1D numpy array in original musicnn feature space (for naming)
        """
        if hybrid_vec is None or len(hybrid_vec) == 0:
            return None

        hybrid_vec = np.array(hybrid_vec).flatten()
        # Use provided dims or fall back to model attributes
        n_musicnn_pca = musicnn_output_dims if musicnn_output_dims is not None else (self.m_pca.n_components_ if self.m_pca else self.n_musicnn_dims)

        # Split hybrid vector into musicnn portion for inversion (maest portion not needed for naming)
        musicnn_portion = hybrid_vec[:n_musicnn_pca]

        # Un-weight (reverse the weight scaling applied during hybrid blending).
        # Uses the evolved weight passed at construction (falls back to the config
        # constant for backward compatibility).
        musicnn_unweighted = musicnn_portion / self.weight_musicnn if self.weight_musicnn > 0 else musicnn_portion

        # Inverse PCA for musicnn (to get back to scaled space)
        if self.m_pca is not None:
            musicnn_scaled = self.m_pca.inverse_transform(musicnn_unweighted.reshape(1, -1)).flatten()
        else:
            musicnn_scaled = musicnn_unweighted

        # Inverse scale for musicnn (to get back to original space)
        if self.m_scaler is not None:
            musicnn_original = self.m_scaler.inverse_transform(musicnn_scaled.reshape(1, -1)).flatten()
        else:
            musicnn_original = musicnn_scaled

        # Truncate to original musicnn dimensions if needed
        if len(musicnn_original) > self.n_musicnn_dims:
            musicnn_original = musicnn_original[:self.n_musicnn_dims]

        return musicnn_original


def _pca_reduce(data, n_components, log_prefix, stream_name=""):
    """PCA-reduce data to n_components. Returns (scaler, pca_model, reduced_data).
    Returns (None, None, data) if n_components >= data.shape[1].
    """

    if data is None or data.shape[0] == 0:
        return None, None, None

    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    effective_n = min(n_components, data.shape[1], data.shape[0])

    if effective_n <= 0 or effective_n >= data.shape[1]:
        return scaler, None, scaled

    pca = PCA(n_components=effective_n)
    reduced = pca.fit_transform(scaled)

    return scaler, pca, reduced


def _prepare_and_scale_data(X_feat, X_embed, use_embeddings):
    """Selects the data source for clustering (features or embeddings) and scales it."""
    data_source = X_embed if use_embeddings else X_feat
    if data_source is None or data_source.shape[0] == 0:
        return None, None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_source)
    return scaled_data, scaler

# --- Step 2: Evolutionary Parameter Generation ---

def _mutate_param(value, min_val, max_val, delta, is_float=False):
    """Applies a random mutation to a single parameter value."""
    if is_float:
        mutation = random.uniform(-delta, delta)
        new_value = value + mutation
    else:
        int_delta = max(1, int(delta))
        mutation = random.randint(-int_delta, int_delta)
        new_value = value + mutation
    new_value = np.clip(new_value, min_val, max_val)
    return int(new_value) if not is_float else new_value

def _generate_evolutionary_parameters(elites, exploitation_prob, mutation_cfg, method, data, *args, clustering_mode='musicnn'):
    """Decides to explore (random params) or exploit (mutate elite params).

    When clustering_mode is 'dual_consensus', filters elites to only those
    tagged with the same mode and generates per-stream PCA params.
    """
    # Filter elites by mode to prevent cross-contamination
    if clustering_mode == 'dual_consensus':
        mode_elites = [e for e in elites if e.get('clustering_mode') == 'dual_consensus']
    else:
        mode_elites = [e for e in elites if e.get('clustering_mode', 'musicnn') in ('musicnn', 'hybrid_blend', None)]

    if mode_elites and random.random() < exploitation_prob:
        chosen_elite = random.choice(mode_elites)
        return _mutate_parameters(chosen_elite, mutation_cfg, method, data, *args, clustering_mode=clustering_mode)
    return _generate_random_parameters(method, data, *args, clustering_mode=clustering_mode)

def _generate_random_parameters(method, data, pca_ranges, num_clust_ranges, db_ranges, gmm_ranges, spec_ranges, *args, clustering_mode='musicnn'):
    """Generates a completely new set of random parameters for clustering.

    For dual_consensus mode, generates per-stream PCA configs via
    _generate_random_dual_params.
    """
    if clustering_mode == 'dual_consensus' and isinstance(data, tuple):
        return _generate_random_dual_params(method, data, pca_ranges, num_clust_ranges,
                                            db_ranges, gmm_ranges, spec_ranges)

    max_pca = min(pca_ranges['components_max'], data.shape[1], data.shape[0] - 1)
    min_pca = pca_ranges['components_min']
    if min_pca > max_pca:
        min_pca = max_pca

    pca_comps = random.randint(min_pca, max_pca) if max_pca >= min_pca and max_pca > 0 else min_pca
    pca_config = {"enabled": pca_comps > 0, "components": pca_comps}

    max_k = data.shape[0]
    method_params = {}

    if method == 'kmeans':
        upper_k = min(num_clust_ranges[1], max_k)
        lower_k = min(num_clust_ranges[0], upper_k)
        if lower_k < 2 and upper_k >= 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        k = random.randint(lower_k, upper_k) if upper_k >= lower_k and upper_k > 0 else lower_k
        method_params = {"n_clusters": k}

    elif method == 'dbscan':
        eps = round(random.uniform(db_ranges['eps_min'], db_ranges['eps_max']), 2)
        min_samples = random.randint(db_ranges['samples_min'], db_ranges['samples_max'])
        method_params = {"eps": eps, "min_samples": min_samples}

    elif method == 'gmm':
        upper_k = min(gmm_ranges['n_components_max'], max_k)
        lower_k = min(gmm_ranges['n_components_min'], upper_k)
        if lower_k < 2 and upper_k >= 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        n_comp = random.randint(lower_k, upper_k) if upper_k >= lower_k and upper_k > 0 else lower_k
        method_params = {"n_components": n_comp}

    elif method == 'spectral':
        upper_k = min(spec_ranges['n_clusters_max'], data.shape[0] - 1)
        lower_k = spec_ranges['n_clusters_min']
        if lower_k < 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        n_clust = random.randint(lower_k, upper_k) if upper_k >= lower_k else lower_k
        method_params = {"n_clusters": n_clust, "random_state": random.randint(0, 10000)}

    result = {"pca_config": pca_config, "clustering_method_config": {"method": method, "params": method_params},
              "clustering_mode": clustering_mode}
    # Evolvable fusion weights for hybrid_blend mode (sum to 1.0, in [0.1, 0.9]).
    # dual_consensus is handled by _generate_random_dual_params via the early return above.
    if clustering_mode == 'hybrid_blend':
        w_mn = random.uniform(0.1, 0.9)
        w_ma = 1.0 - w_mn
        result['hybrid_weights'] = {'musicnn': w_mn, 'maest': w_ma}
    return result


def _generate_random_dual_params(method, data_tuple, pca_ranges, num_clust_ranges, db_ranges, gmm_ranges, spec_ranges):
    """Generates random per-stream PCA parameters for dual consensus.

    Each stream gets its own PCA components clamped to its dimension,
    enabling independent exploration of the musicnn (58-dim) and MAEST
    (469-dim) reduction spaces.
    """
    X_mn, X_ma = data_tuple

    def _rand_pca(data):
        max_pca = min(pca_ranges['components_max'], data.shape[1], data.shape[0] - 1)
        min_pca = pca_ranges['components_min']
        if min_pca > max_pca:
            min_pca = max_pca
        comps = random.randint(min_pca, max_pca) if max_pca >= min_pca and max_pca > 0 else min_pca
        return {"enabled": comps > 0, "components": comps}

    musicnn_pca = _rand_pca(X_mn)
    maest_pca = _rand_pca(X_ma)

    # Clustering method uses the smaller stream's track count for n_clusters bounds
    max_k = X_mn.shape[0]
    method_params = {}

    if method == 'kmeans':
        upper_k = min(num_clust_ranges[1], max_k)
        lower_k = min(num_clust_ranges[0], upper_k)
        if lower_k < 2 and upper_k >= 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        k = random.randint(lower_k, upper_k) if upper_k >= lower_k and upper_k > 0 else lower_k
        method_params = {"n_clusters": k}

    elif method == 'dbscan':
        eps = round(random.uniform(db_ranges['eps_min'], db_ranges['eps_max']), 2)
        min_samples = random.randint(db_ranges['samples_min'], db_ranges['samples_max'])
        method_params = {"eps": eps, "min_samples": min_samples}

    elif method == 'gmm':
        upper_k = min(gmm_ranges['n_components_max'], max_k)
        lower_k = min(gmm_ranges['n_components_min'], upper_k)
        if lower_k < 2 and upper_k >= 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        n_comp = random.randint(lower_k, upper_k) if upper_k >= lower_k and upper_k > 0 else lower_k
        method_params = {"n_components": n_comp}

    elif method == 'spectral':
        upper_k = min(spec_ranges['n_clusters_max'], max_k - 1)
        lower_k = spec_ranges['n_clusters_min']
        if lower_k < 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        n_clust = random.randint(lower_k, upper_k) if upper_k >= lower_k else lower_k
        method_params = {"n_clusters": n_clust, "random_state": random.randint(0, 10000)}

    # Evolvable fusion weights for dual_consensus mode (sum to 1.0, in [0.1, 0.9]).
    w_mn = random.uniform(0.1, 0.9)
    w_ma = 1.0 - w_mn
    return {
        "musicnn_pca": musicnn_pca,
        "maest_pca": maest_pca,
        "pca_config": {"enabled": False, "components": 0},  # Disabled — per-stream PCAs are embedded
        "clustering_method_config": {"method": method, "params": method_params},
        "clustering_mode": "dual_consensus",
        "hybrid_weights": {'musicnn': w_mn, 'maest': w_ma},
    }

def _mutate_parameters(elite_params, mutation_cfg, method, data, pca_ranges, num_clust_ranges, db_ranges, gmm_ranges, spec_ranges, *args, clustering_mode='musicnn'):
    """Takes an elite parameter set and applies small random changes.

    For dual_consensus elites, mutates musicnn_pca and maest_pca
    independently within their respective dimension bounds.
    """
    clustering_mode = elite_params.get('clustering_mode', 'musicnn')

    if clustering_mode == 'dual_consensus':
        return _mutate_dual_parameters(elite_params, mutation_cfg, method, data,
                                       pca_ranges, num_clust_ranges, db_ranges, gmm_ranges, spec_ranges)

    elite_pca_cfg = elite_params['pca_config']
    elite_method_cfg = elite_params['clustering_method_config']

    max_pca = min(pca_ranges['components_max'], data.shape[1], data.shape[0] - 1)
    min_pca = pca_ranges['components_min']
    if min_pca > max_pca:
        min_pca = max_pca
    mutated_pca_comps = _mutate_param(elite_pca_cfg.get('components', 0), min_pca, max_pca, mutation_cfg.get('int_abs_delta', 2))
    pca_config = {"enabled": mutated_pca_comps > 0, "components": mutated_pca_comps}

    max_k = data.shape[0]
    method_params = {}

    if method == 'kmeans':
        upper_k = min(num_clust_ranges[1], max_k)
        lower_k = min(num_clust_ranges[0], upper_k)
        k = _mutate_param(elite_method_cfg['params']['n_clusters'], lower_k, upper_k, mutation_cfg.get('int_abs_delta', 2))
        method_params = {"n_clusters": k}
    elif method == 'dbscan':
        mutated_eps = _mutate_param(elite_method_cfg['params']['eps'], db_ranges['eps_min'], db_ranges['eps_max'], mutation_cfg.get('float_abs_delta', 0.1), is_float=True)
        mutated_min_samples = _mutate_param(elite_method_cfg['params']['min_samples'], db_ranges['samples_min'], db_ranges['samples_max'], mutation_cfg.get('int_abs_delta', 2))
        method_params = {"eps": mutated_eps, "min_samples": mutated_min_samples}
    elif method == 'gmm':
        upper_k = min(gmm_ranges['n_components_max'], max_k)
        lower_k = min(gmm_ranges['n_components_min'], upper_k)
        n_comp = _mutate_param(elite_method_cfg['params']['n_components'], lower_k, upper_k, mutation_cfg.get('int_abs_delta', 2))
        method_params = {"n_components": n_comp}
    elif method == 'spectral':
        upper_k = min(spec_ranges['n_clusters_max'], max_k - 1)
        lower_k = spec_ranges['n_clusters_min']
        if lower_k < 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        n_clust = _mutate_param(elite_method_cfg['params']['n_clusters'], lower_k, upper_k, mutation_cfg.get('int_abs_delta', 2))
        elite_random_state = elite_method_cfg['params'].get("random_state", random.randint(0, 10000))
        mutated_random_state = _mutate_param(elite_random_state, 0, 10000, mutation_cfg.get("int_abs_delta", 100))
        method_params = {"n_clusters": n_clust, "random_state": mutated_random_state}

    result = {"pca_config": pca_config, "clustering_method_config": {"method": method, "params": method_params},
              "clustering_mode": clustering_mode}
    # Mutate evolvable fusion weights when present (hybrid_blend mode).
    # Uses _mutate_param with the float delta, then renormalizes to sum to 1.0
    # and clamps each weight to [0.05, 0.95] to keep them meaningful.
    if 'hybrid_weights' in elite_params:
        elite_w = elite_params['hybrid_weights']
        elite_mn = float(elite_w.get('musicnn', HYBRID_WEIGHT_MUSICNN))
        elite_ma = float(elite_w.get('maest', HYBRID_WEIGHT_MAEST))
        float_delta = mutation_cfg.get('float_abs_delta', 0.1)
        new_mn = _mutate_param(elite_mn, 0.05, 0.95, float_delta, is_float=True)
        new_ma = _mutate_param(elite_ma, 0.05, 0.95, float_delta, is_float=True)
        total = new_mn + new_ma
        if total > 0:
            new_mn = new_mn / total
            new_ma = new_ma / total
        # Re-clamp after renormalization and adjust to keep sum == 1.0
        new_mn = min(0.95, max(0.05, new_mn))
        new_ma = 1.0 - new_mn
        result['hybrid_weights'] = {'musicnn': new_mn, 'maest': new_ma}
    return result


def _mutate_dual_parameters(elite_params, mutation_cfg, method, data_tuple, pca_ranges, num_clust_ranges, db_ranges, gmm_ranges, spec_ranges):
    """Mutates a dual_consensus elite's per-stream PCA configs independently."""
    X_mn, X_ma = data_tuple
    elite_method_cfg = elite_params['clustering_method_config']
    delta = mutation_cfg.get('int_abs_delta', 2)

    # Mutate musicnn PCA
    elite_mn = elite_params.get('musicnn_pca', {'enabled': True, 'components': 20})
    max_mn = min(pca_ranges['components_max'], X_mn.shape[1], X_mn.shape[0] - 1)
    min_mn = pca_ranges['components_min']
    if min_mn > max_mn: min_mn = max_mn
    mutated_mn = _mutate_param(elite_mn.get('components', 20), min_mn, max_mn, delta)
    musicnn_pca = {"enabled": mutated_mn > 0, "components": mutated_mn}

    # Mutate maest PCA
    elite_ma = elite_params.get('maest_pca', {'enabled': True, 'components': 60})
    max_ma = min(pca_ranges['components_max'], X_ma.shape[1], X_ma.shape[0] - 1)
    min_ma = pca_ranges['components_min']
    if min_ma > max_ma: min_ma = max_ma
    mutated_ma = _mutate_param(elite_ma.get('components', 60), min_ma, max_ma, delta)
    maest_pca = {"enabled": mutated_ma > 0, "components": mutated_ma}

    # Mutate clustering method
    max_k = X_mn.shape[0]
    method_params = {}

    if method == 'kmeans':
        upper_k = min(num_clust_ranges[1], max_k)
        lower_k = min(num_clust_ranges[0], upper_k)
        k = _mutate_param(elite_method_cfg['params']['n_clusters'], lower_k, upper_k, delta)
        method_params = {"n_clusters": k}
    elif method == 'dbscan':
        mutated_eps = _mutate_param(elite_method_cfg['params']['eps'], db_ranges['eps_min'], db_ranges['eps_max'], mutation_cfg.get('float_abs_delta', 0.1), is_float=True)
        mutated_min_samples = _mutate_param(elite_method_cfg['params']['min_samples'], db_ranges['samples_min'], db_ranges['samples_max'], delta)
        method_params = {"eps": mutated_eps, "min_samples": mutated_min_samples}
    elif method == 'gmm':
        upper_k = min(gmm_ranges['n_components_max'], max_k)
        lower_k = min(gmm_ranges['n_components_min'], upper_k)
        n_comp = _mutate_param(elite_method_cfg['params']['n_components'], lower_k, upper_k, delta)
        method_params = {"n_components": n_comp}
    elif method == 'spectral':
        upper_k = min(spec_ranges['n_clusters_max'], max_k - 1)
        lower_k = spec_ranges['n_clusters_min']
        if lower_k < 2: lower_k = 2
        if upper_k < lower_k: upper_k = lower_k
        n_clust = _mutate_param(elite_method_cfg['params']['n_clusters'], lower_k, upper_k, delta)
        elite_random_state = elite_method_cfg['params'].get("random_state", random.randint(0, 10000))
        mutated_random_state = _mutate_param(elite_random_state, 0, 10000, mutation_cfg.get("int_abs_delta", 100))
        method_params = {"n_clusters": n_clust, "random_state": mutated_random_state}

    result = {
        "musicnn_pca": musicnn_pca,
        "maest_pca": maest_pca,
        "pca_config": {"enabled": False, "components": 0},
        "clustering_method_config": {"method": method, "params": method_params},
        "clustering_mode": "dual_consensus",
    }
    # Mutate evolvable fusion weights when present (dual_consensus mode).
    # Uses _mutate_param with the float delta, then renormalizes to sum to 1.0
    # and clamps each weight to [0.05, 0.95] to keep them meaningful.
    if 'hybrid_weights' in elite_params:
        elite_w = elite_params['hybrid_weights']
        elite_mn = float(elite_w.get('musicnn', HYBRID_WEIGHT_MUSICNN))
        elite_ma = float(elite_w.get('maest', HYBRID_WEIGHT_MAEST))
        float_delta = mutation_cfg.get('float_abs_delta', 0.1)
        new_mn = _mutate_param(elite_mn, 0.05, 0.95, float_delta, is_float=True)
        new_ma = _mutate_param(elite_ma, 0.05, 0.95, float_delta, is_float=True)
        total = new_mn + new_ma
        if total > 0:
            new_mn = new_mn / total
            new_ma = new_ma / total
        # Re-clamp after renormalization and adjust to keep sum == 1.0
        new_mn = min(0.95, max(0.05, new_mn))
        new_ma = 1.0 - new_mn
        result['hybrid_weights'] = {'musicnn': new_mn, 'maest': new_ma}
    return result

# --- Step 3 & 4: Apply Models ---

def _apply_clustering_model(data, method_config, log_prefix, run_idx):
    """Initializes and fits the specified clustering model (with optional GPU acceleration)."""
    method = method_config['method']
    params = method_config['params']
    model = None
    try:
        # Validate parameters before creating model
        if method == 'kmeans':
            if params.get('n_clusters', 0) < 2:
                return None, None, None
        elif method == 'gmm':
            if params.get('n_components', 0) < 2 or params['n_components'] > data.shape[0]:
                return None, None, None
        elif method == 'spectral':
            if params.get('n_clusters', 0) < 2 or params['n_clusters'] >= data.shape[0]:
                return None, None, None

        # Use GPU clustering if enabled and available
        use_gpu = USE_GPU_CLUSTERING and GPU_CLUSTERING_AVAILABLE

        if use_gpu:
            try:
                model = get_clustering_model(method, params, use_gpu=True)
                labels = model.fit_predict(data)
                logger.debug(f"{log_prefix} Iteration {run_idx}: GPU clustering used for {method}")
            except Exception as e:
                logger.warning(f"{log_prefix} GPU clustering failed, falling back to CPU: {e}")
                use_gpu = False

        # Use CPU clustering (either by choice or as fallback)
        if not use_gpu:
            if method == 'kmeans':
                model = KMeans(n_clusters=params['n_clusters'], init='k-means++', n_init=10)
            elif method == 'dbscan':
                model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
            elif method == 'gmm':
                model = GaussianMixture(
                    n_components=params['n_components'],
                    covariance_type=GMM_COVARIANCE_TYPE,
                    init_params='k-means++',
                    n_init=10,
                    random_state=None,
                    reg_covar=1e-4
                )
            elif method == 'spectral':
                model = SpectralClustering(
                    n_clusters=params['n_clusters'],
                    assign_labels='kmeans',
                    affinity='nearest_neighbors',
                    n_neighbors=SPECTRAL_N_NEIGHBORS,
                    random_state=params.get("random_state"),
                    n_init=10,
                    verbose=False
                )
            else:
                raise ValueError(f"Unsupported clustering method: {method}")

            labels = model.fit_predict(data)

        # Extract cluster centers
        centers = {}
        if hasattr(model, 'cluster_centers_') and model.cluster_centers_ is not None:
            centers = {i: center for i, center in enumerate(model.cluster_centers_)}
        elif hasattr(model, 'means_') and model.means_ is not None:
            centers = {i: mean for i, mean in enumerate(model.means_)}
        else: # Fallback for DBSCAN and Spectral
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            for label in unique_labels:
                cluster_points = data[labels == label]
                if cluster_points.shape[0] > 0:
                    centers[label] = cluster_points.mean(axis=0)

        return labels, centers, model

    except Exception:
        logger.error(f"{log_prefix} Iteration {run_idx}: Clustering model failed for method {method}", exc_info=True)
        return None, None, None


def _get_feature_centroid_for_embedding_cluster(label_id, labels, X_feat_orig):
    """
    When clustering on embeddings, this calculates a representative centroid
    in the original feature space for naming and analysis.
    """
    cluster_indices = np.where(labels == label_id)[0]
    if len(cluster_indices) == 0:
        return None

    feature_vectors_in_cluster = X_feat_orig[cluster_indices]
    feature_centroid = np.mean(feature_vectors_in_cluster, axis=0)
    return feature_centroid

# --- Step 5 & 6: Formatting and Scoring ---

# Map of clustering_mode -> LN stats dicts for Z-score normalization
_LN_STATS_BY_MODE = {
    'musicnn': {'diversity': LN_MOOD_DIVERSITY_STATS, 'purity': LN_MOOD_PURITY_STATS},
    'maest':   {'diversity': LN_MAEST_GENRE_DIVERSITY_STATS, 'purity': LN_MAEST_GENRE_PURITY_STATS},
    'hybrid_blend': {'diversity': LN_HYBRID_MOOD_DIVERSITY_STATS, 'purity': LN_HYBRID_MOOD_PURITY_STATS},
    'dual_consensus': {'diversity': LN_HYBRID_MOOD_DIVERSITY_STATS, 'purity': LN_HYBRID_MOOD_PURITY_STATS},
}


def _resolve_ln_stats(use_embeddings, clustering_mode):
    """Return (diversity_stats, purity_stats) dicts for the given mode.

    Embedding-based clustering uses its own stats regardless of mode.
    Falls back to musicnn stats when the mode-specific stats are dummy
    (mean == sd, a sign they haven't been calibrated).
    """
    if use_embeddings:
        return LN_MOOD_DIVERSITY_EMBEDING_STATS, LN_MOOD_PURITY_EMBEDING_STATS

    mode_key = clustering_mode if clustering_mode in _LN_STATS_BY_MODE else 'musicnn'
    stats = _LN_STATS_BY_MODE[mode_key]
    div_stats, pur_stats = stats['diversity'], stats['purity']

    # Detect uncalibrated dummy values (mean ≈ sd from copy-paste)
    if abs(div_stats.get('mean', 0) - div_stats.get('sd', 1)) < 0.01:
        div_stats = LN_MOOD_DIVERSITY_STATS  # fall back to musicnn
    if abs(pur_stats.get('mean', 0) - pur_stats.get('sd', 1)) < 0.01:
        pur_stats = LN_MOOD_PURITY_STATS

    return div_stats, pur_stats


def _calibrate_ln_stats(num_samples=2000, num_fast_iterations=20, clustering_mode='musicnn'):
    """Run fast calibration pass to estimate LN_*_STATS for the given mode.

    Samples tracks, runs a few KMeans clusterings with random parameters,
    records raw mood diversity and purity distributions, and returns
    calibrated (diversity_stats, purity_stats) dicts with mean/sd.

    Returns None when there isn't enough data.
    """
    try:
        # Fetch sample data via score_vector-compatible query
        from database import get_db
        from psycopg2.extras import DictCursor
        conn = get_db()
        with conn.cursor(cursor_factory=DictCursor) as cur:
            if clustering_mode == 'maest':
                cur.execute(
                    "SELECT item_id, title, author, tempo, key, scale, energy, "
                    "maest_mood_vector AS mood_vector, other_features, album, album_artist, year, rating, file_path "
                    "FROM score WHERE maest_mood_vector IS NOT NULL AND maest_mood_vector <> '' LIMIT %s",
                    (num_samples,)
                )
            else:
                cur.execute(
                    "SELECT item_id, title, author, tempo, key, scale, energy, "
                    "mood_vector, other_features, album, album_artist, year, rating, file_path "
                    "FROM score WHERE mood_vector IS NOT NULL AND mood_vector <> '' LIMIT %s",
                    (num_samples,)
                )
            rows = cur.fetchall()

        # Build feature vectors — query already aliases maest_mood_vector AS mood_vector
        mood_labels = MAEST_MOOD_LABELS if clustering_mode == 'maest' else MOOD_LABELS
        valid_tracks, feat_list = [], []
        for row_data in (dict(r) for r in rows if r):
            try:
                vec = score_vector(row_data, mood_labels, OTHER_FEATURE_LABELS, mood_column='mood_vector')
                valid_tracks.append(row_data)
                feat_list.append(vec)
            except Exception:
                continue

        if len(valid_tracks) < 200:
            return None

        X = np.array(feat_list)
        if len(valid_tracks) > num_samples:
            idx = np.random.choice(len(valid_tracks), num_samples, replace=False)
            X = X[idx]

        # Run fast clusterings and record raw scores
        raw_diversities, raw_purities = [], []
        scaled = StandardScaler().fit_transform(X)
        max_k = min(100, len(scaled) // 10)

        for _ in range(num_fast_iterations):
            k = np.random.randint(max(10, max_k // 2), max_k + 1)
            km = KMeans(n_clusters=k, n_init=1, max_iter=100, random_state=np.random.randint(10000))
            labels = km.fit_predict(scaled)

            centroids = {}
            for cid in set(labels):
                if cid == -1:
                    continue
                mask = labels == cid
                centroids[cid] = X[mask].mean(axis=0)

            # Diversity: unique predominant moods across all clusters
            unique_moods = {}
            for cid, center in centroids.items():
                mood_scores = {mood_labels[i]: center[2 + i] for i in range(len(mood_labels)) if center[2 + i] > 0.01}
                if mood_scores:
                    top_mood = max(mood_scores, key=mood_scores.get)
                    unique_moods[top_mood] = max(unique_moods.get(top_mood, 0), mood_scores[top_mood])

            raw_diversity = sum(unique_moods.values())
            raw_diversities.append(raw_diversity)

            # Purity: per-cluster sum of top-mood scores
            total_purity = 0.0
            for cid, center in centroids.items():
                mood_scores = {mood_labels[i]: center[2 + i] for i in range(len(mood_labels)) if center[2 + i] > 0.01}
                if mood_scores:
                    total_purity += sum(sorted(mood_scores.values(), reverse=True)[:3])
            raw_purities.append(total_purity)

        if len(raw_diversities) < 5:
            return None

        ln_div = np.log1p(raw_diversities)
        ln_pur = np.log1p(raw_purities)

        return (
            {'min': float(ln_div.min()), 'max': float(ln_div.max()),
             'mean': float(ln_div.mean()), 'sd': float(ln_div.std())},
            {'min': float(ln_pur.min()), 'max': float(ln_pur.max()),
             'mean': float(ln_pur.mean()), 'sd': float(ln_pur.std())},
        )
    except Exception as e:
        logger.warning(f"LN stats calibration failed: {e}")
        return None

def _format_and_score_iteration_result(
    labels, valid_tracks, X_feat_orig, data_for_metrics,
    centers, model, pca, scaler, active_moods,
    params, max_songs_per_cluster, run_idx, use_embeddings, score_weights, log_prefix, clustering_mode='musicnn'):
    """
    Packages all results from the iteration into a dictionary and calculates the final fitness score.
    This version includes the advanced filtering and scoring logic.
    """
    if labels is None:
        return {"fitness_score": -1.0}

    # --- 1. Filter clusters to create final playlists ---
    raw_distances = np.full(len(valid_tracks), np.inf)
    if len(set(labels) - {-1}) > 0:
        for label_id in set(labels):
            if label_id == -1: continue
            indices = np.where(labels == label_id)[0]
            if len(indices) > 0 and label_id in centers:
                cluster_center = centers[label_id]
                points = data_for_metrics[indices]
                distances = np.linalg.norm(points - cluster_center, axis=1)
                raw_distances[indices] = distances

    max_dist_val = raw_distances[raw_distances != np.inf].max() if np.any(raw_distances != np.inf) else 1.0
    if max_dist_val == 0: max_dist_val = 1.0
    normalized_distances = raw_distances / max_dist_val

    track_info_list = [{"row": valid_tracks[i], "label": labels[i], "distance": normalized_distances[i]} for i in range(len(valid_tracks))]

    filtered_clusters = defaultdict(list)
    for cid in set(labels):
        if cid == -1: continue
        cluster_tracks_info = [t_info for t_info in track_info_list if t_info["label"] == cid and t_info["distance"] <= MAX_DISTANCE]
        if not cluster_tracks_info: continue

        cluster_tracks_info.sort(key=lambda x: x["distance"])
        # Track per-artist counts using a normalized author key. Treat MAX_SONGS_PER_ARTIST <= 0
        # or None as DISABLED (no cap), consistent with other modules (path_manager/ivf_manager).
        count_per_artist = defaultdict(int)
        selected_tracks_for_playlist = []
        for t_item_info in cluster_tracks_info:
            author = t_item_info["row"].get("author")
            author_norm = (author or "").strip().lower()

            # If MAX_SONGS_PER_ARTIST is not configured or <= 0, disable per-artist cap.
            if MAX_SONGS_PER_ARTIST is None or MAX_SONGS_PER_ARTIST <= 0:
                allowed_by_artist = True
            else:
                allowed_by_artist = count_per_artist[author_norm] < MAX_SONGS_PER_ARTIST

            if allowed_by_artist:
                selected_tracks_for_playlist.append(t_item_info)
                count_per_artist[author_norm] += 1

            if max_songs_per_cluster > 0 and len(selected_tracks_for_playlist) >= max_songs_per_cluster:
                break

        for t_item_info_final in selected_tracks_for_playlist:
            item_id_val, title_val, author_val = t_item_info_final["row"]["item_id"], t_item_info_final["row"]["title"], t_item_info_final["row"]["author"]
            filtered_clusters[cid].append((item_id_val, title_val, author_val))

    # --- 2. Format final playlists and centroids ---
    named_playlists, playlist_centroids = {}, {}
    # *** NEW: Map final playlist names to their centroid vectors for Top-N selection ***
    playlist_to_centroid_vector_map = {}
    unique_predominant_mood_scores = {}
    unique_predominant_other_feature_scores = {}
    item_id_to_song_index_map = {track_data['item_id']: i for i, track_data in enumerate(valid_tracks)}

    for label_id, songs_list in filtered_clusters.items():
        if songs_list and label_id in centers:
            center_vec = centers[label_id] # This is the vector in the clustered space

            # Check if this is hybrid mode and we need to invert the centroid
            hybrid_pca_models = params.get('_hybrid_pca_models')
            is_hybrid_mode = bool(hybrid_pca_models)

            if use_embeddings:
                feature_centroid_vec = _get_feature_centroid_for_embedding_cluster(label_id, labels, X_feat_orig)
                if feature_centroid_vec is None: continue
                name, centroid_details = _name_cluster(feature_centroid_vec, None, False, active_moods, None)
            elif is_hybrid_mode:
                # Hybrid mode: invert hybrid-space centroid back to musicnn feature space for naming
                # Extract hybrid PCA models
                m_model = hybrid_pca_models.get('musicnn', {})
                a_model = hybrid_pca_models.get('maest', {})
                m_scaler_h = m_model.get('scaler')
                m_pca_h = m_model.get('pca')
                a_scaler_h = a_model.get('scaler')
                a_pca_h = a_model.get('pca')

                # Get original musicnn dimensionality (58: 2 + 55 mood labels + other features)
                n_musicnn_dims = 2 + len(MOOD_LABELS) + len(OTHER_FEATURE_LABELS)

                # Create HybridScaler and invert
                # Pass the evolved fusion weights so the un-weighting step in
                # inverse_transform matches the weighting applied during blending.
                hybrid_weights = params.get('_hybrid_weights', {})
                w_mn_h = hybrid_weights.get('musicnn', HYBRID_WEIGHT_MUSICNN)
                w_ma_h = hybrid_weights.get('maest', HYBRID_WEIGHT_MAEST)
                musicnn_out_dims = m_model.get('output_dims')
                maest_out_dims = a_model.get('output_dims')
                hybrid_scaler = HybridScaler(m_scaler_h, m_pca_h, a_scaler_h, a_pca_h, n_musicnn_dims,
                                             weight_musicnn=w_mn_h, weight_maest=w_ma_h)
                inverted_vec = hybrid_scaler.inverse_transform(center_vec, musicnn_out_dims, maest_out_dims)
                if inverted_vec is None:
                    logger.warning(f"{log_prefix} Iteration {run_idx}: Failed to invert hybrid centroid for cluster {label_id}. Skipping.")
                    continue

                # Name using the inverted vector (no additional scaler/pca needed)
                name, centroid_details = _name_cluster(inverted_vec, None, False, active_moods, None)
            else:
                name, centroid_details = _name_cluster(center_vec, pca, params['pca_config']['enabled'], active_moods, scaler)

            temp_name, suffix = name, 1
            while temp_name in named_playlists:
                temp_name = f"{name}_{suffix}"
                suffix += 1

            named_playlists[temp_name] = songs_list
            playlist_centroids[temp_name] = centroid_details
            # *** NEW: Store the mapping from the final unique name to the centroid vector ***
            playlist_to_centroid_vector_map[temp_name] = center_vec

            if centroid_details and any(mood in active_moods for mood in centroid_details.keys()):
                predominant_mood_key = max((k for k in centroid_details if k in MOOD_LABELS), key=centroid_details.get, default=None)
                if predominant_mood_key:
                    current_mood_score = centroid_details.get(predominant_mood_key, 0.0)
                    unique_predominant_mood_scores[predominant_mood_key] = max(unique_predominant_mood_scores.get(predominant_mood_key, 0.0), current_mood_score)

            centroid_other_features = {lk: centroid_details.get(lk, 0.0) for lk in OTHER_FEATURE_LABELS if lk in centroid_details}
            if centroid_other_features:
                predominant_other_key = max(centroid_other_features, key=centroid_other_features.get, default=None)
                if predominant_other_key and centroid_other_features[predominant_other_key] > OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY:
                     unique_predominant_other_feature_scores[predominant_other_key] = max(unique_predominant_other_feature_scores.get(predominant_other_key, 0.0), centroid_other_features[predominant_other_key])

    # --- 3. Calculate All Metrics ---
    metrics = {"silhouette": 0.0, "davies_bouldin": 0.0, "calinski_harabasz": 0.0, "mood_diversity": 0.0, "mood_purity": 0.0, "other_feature_diversity": 0.0, "other_feature_purity": 0.0}
    num_clusters = len(named_playlists)

    if num_clusters >= 2 and num_clusters < data_for_metrics.shape[0]:
        if score_weights.get('silhouette', 0) > 0:
            try: metrics['silhouette'] = (silhouette_score(data_for_metrics, labels) + 1) / 2.0
            except ValueError: pass
        if score_weights.get('davies_bouldin', 0) > 0:
            try: metrics['davies_bouldin'] = 1.0 / (1.0 + davies_bouldin_score(data_for_metrics, labels))
            except ValueError: pass
        if score_weights.get('calinski_harabasz', 0) > 0:
            try: metrics['calinski_harabasz'] = 1.0 - np.exp(-calinski_harabasz_score(data_for_metrics, labels) / 500.0)
            except ValueError: pass

    raw_mood_diversity_score = sum(unique_predominant_mood_scores.values())
    ln_mood_diversity = np.log1p(raw_mood_diversity_score)
    _diversity_stats, _purity_stats = _resolve_ln_stats(use_embeddings, clustering_mode)
    mean_div, sd_div = _diversity_stats.get("mean"), _diversity_stats.get("sd")
    if mean_div is not None and sd_div is not None and sd_div > 1e-9:
        metrics['mood_diversity'] = (ln_mood_diversity - mean_div) / sd_div

    raw_other_diversity_score = sum(unique_predominant_other_feature_scores.values())
    ln_other_diversity = np.log1p(raw_other_diversity_score)
    other_div_stats = LN_OTHER_FEATURES_DIVERSITY_STATS
    mean_other_div, sd_other_div = other_div_stats.get("mean"), other_div_stats.get("sd")
    if mean_other_div is not None and sd_other_div is not None and sd_other_div > 1e-9:
        metrics['other_feature_diversity'] = (ln_other_diversity - mean_other_div) / sd_other_div

    all_playlist_purities = []
    if named_playlists:
        for name, songs in named_playlists.items():
            centroid_data = playlist_centroids.get(name)
            if not centroid_data or not songs: continue

            sorted_moods = sorted([(m,s) for m,s in centroid_data.items() if m in MOOD_LABELS], key=lambda item: item[1], reverse=True)
            top_moods = [m for m, s in sorted_moods[:TOP_K_MOODS_FOR_PURITY_CALCULATION] if s > 0.01]
            if not top_moods: continue

            song_purity_scores = []
            for item_id, _, _ in songs:
                song_idx = item_id_to_song_index_map.get(item_id)
                if song_idx is not None and song_idx < X_feat_orig.shape[0]:
                    song_feat_vec = X_feat_orig[song_idx]
                    max_score_for_song = 0.0
                    for mood in top_moods:
                        try:
                            mood_idx = active_moods.index(mood)
                            if 2 + mood_idx < song_feat_vec.shape[0]:
                                song_score = song_feat_vec[2 + mood_idx]
                                if song_score > max_score_for_song:
                                    max_score_for_song = song_score
                        except ValueError:
                            continue
                    if max_score_for_song > 0:
                        song_purity_scores.append(max_score_for_song)
            if song_purity_scores:
                all_playlist_purities.append(sum(song_purity_scores))

    raw_mood_purity = sum(all_playlist_purities)
    ln_mood_purity = np.log1p(raw_mood_purity)
    mean_pur, sd_pur = _purity_stats.get("mean"), _purity_stats.get("sd")
    if mean_pur is not None and sd_pur is not None and sd_pur > 1e-9:
        metrics['mood_purity'] = (ln_mood_purity - mean_pur) / sd_pur

    all_other_feature_purities = []
    if named_playlists:
        for name, songs in named_playlists.items():
            centroid_data = playlist_centroids.get(name)
            if not centroid_data or not songs: continue

            other_features = {k: v for k, v in centroid_data.items() if k in OTHER_FEATURE_LABELS}
            if not other_features: continue

            predominant_other = max(other_features, key=other_features.get, default=None)
            if not predominant_other or other_features[predominant_other] < OTHER_FEATURE_PREDOMINANCE_THRESHOLD_FOR_PURITY:
                continue

            try:
                feature_idx = OTHER_FEATURE_LABELS.index(predominant_other)
                song_purity_scores = []
                for item_id, _, _ in songs:
                    song_idx = item_id_to_song_index_map.get(item_id)
                    if song_idx is not None and song_idx < X_feat_orig.shape[0]:
                        song_feat_vec = X_feat_orig[song_idx]
                        other_features_start_idx = 2 + len(active_moods)
                        if other_features_start_idx + feature_idx < song_feat_vec.shape[0]:
                            song_score = song_feat_vec[other_features_start_idx + feature_idx]
                            song_purity_scores.append(song_score)
                if song_purity_scores:
                    all_other_feature_purities.append(sum(song_purity_scores))
            except ValueError:
                continue

    raw_other_purity = sum(all_other_feature_purities)
    ln_other_purity = np.log1p(raw_other_purity)
    other_purity_stats = LN_OTHER_FEATURES_PURITY_STATS
    mean_other_pur, sd_other_pur = other_purity_stats.get("mean"), other_purity_stats.get("sd")
    if mean_other_pur is not None and sd_other_pur is not None and sd_other_pur > 1e-9:
        metrics['other_feature_purity'] = (ln_other_purity - mean_other_pur) / sd_other_pur

    # --- 4. Calculate Final Score ---
    final_score = sum(score_weights.get(k, 0) * v for k, v in metrics.items())

    log_message = (
        f"{log_prefix} Iteration {run_idx}: Scores -> "
        f"MoodDiv: {metrics['mood_diversity']:.2f} (raw: {raw_mood_diversity_score:.2f}), "
        f"MoodPur: {metrics['mood_purity']:.2f} (raw: {raw_mood_purity:.2f}), "
        f"OtherFeatDiv: {metrics['other_feature_diversity']:.2f} (raw: {raw_other_diversity_score:.2f}), "
        f"OtherFeatPur: {metrics['other_feature_purity']:.2f} (raw: {raw_other_purity:.2f}), "
        f"Sil: {metrics['silhouette']:.2f}, DB: {metrics['davies_bouldin']:.2f}, CH: {metrics['calinski_harabasz']:.2f} | "
        f"FinalScore: {final_score:.2f}"
    )
    logger.info(log_message)

    # --- 5. Package Final Result ---
    logger.info(f"Run {run_idx}: Created {len(named_playlists)} clusters.")
    for name, songs in named_playlists.items():
        song_titles = [f"'{s[1]}'" for s in songs[:5]]
        log_msg = f"  - Cluster '{name}': {', '.join(song_titles)}"
        if len(songs) > 5:
            log_msg += f", ... and {len(songs) - 5} more."
        logger.info(log_msg)

    return {
        "fitness_score": final_score,
        "named_playlists": named_playlists,
        "playlist_centroids": playlist_centroids,
        "playlist_to_centroid_vector_map": playlist_to_centroid_vector_map, # *** NEW: Return the map ***
        "parameters": {**params, "max_songs_per_cluster": max_songs_per_cluster, "run_id": run_idx},
        "scaler_details": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()} if scaler else None,
        "pca_model_details": {"components": pca.components_.tolist(), "variance": pca.explained_variance_ratio_.tolist()} if pca else None
    }

def _name_cluster(centroid_vector, pca_model, pca_enabled, mood_labels, scaler):
    """Generates a human-readable name for a cluster based on its centroid."""
    # Constants for naming logic
    TOP_MOODS_IN_NAME = 3
    OTHER_FEATURE_THRESHOLD_FOR_NAME = 0.5
    MAX_OTHER_FEATURES_IN_NAME = 2

    # If scaler is None, the vector is already in the original feature space (e.g., from embedding cluster)
    if scaler:
        vec = centroid_vector.reshape(1, -1)
        if pca_enabled and pca_model:
            vec = pca_model.inverse_transform(vec)
        interpreted_vector = scaler.inverse_transform(vec)[0]
    else:
        interpreted_vector = centroid_vector

    # --- Extract features from the vector ---
    tempo_val = interpreted_vector[0]
    mood_values = interpreted_vector[2 : 2 + len(mood_labels)]

    # --- Build Name Components ---
    tempo_label = "Slow" if tempo_val < 0.33 else "Medium" if tempo_val < 0.66 else "Fast"

    if len(mood_values) > 0 and np.sum(mood_values) > 0:
        top_mood_indices = np.argsort(mood_values)[::-1][:TOP_MOODS_IN_NAME]
        mood_names = [mood_labels[i].title() for i in top_mood_indices if i < len(mood_labels) and mood_values[i] > 0.01]
        mood_part = "_".join(mood_names) if mood_names else "Mixed"
    else:
        mood_part = "Mixed"

    base_name = f"{mood_part}_{tempo_label}"

    # --- Extract "Other Features" and add them to the name and details dict ---
    details = {label: float(val) for label, val in zip(mood_labels, mood_values)}
    other_features_start = 2 + len(mood_labels)
    appended_other_features_str = ""
    other_feature_scores_dict = {}

    if len(interpreted_vector) > other_features_start:
        other_feature_values = interpreted_vector[other_features_start:]
        for i, label in enumerate(OTHER_FEATURE_LABELS):
            if i < len(other_feature_values):
                score = float(other_feature_values[i])
                details[label] = score
                other_feature_scores_dict[label] = score

        if other_feature_scores_dict:
            prominent_features = sorted(
                [(feature, score) for feature, score in other_feature_scores_dict.items() if score >= OTHER_FEATURE_THRESHOLD_FOR_NAME],
                key=lambda item: item[1],
                reverse=True
            )
            features_to_add = [feature.title() for feature, score in prominent_features[:MAX_OTHER_FEATURES_IN_NAME]]
            if features_to_add:
                appended_other_features_str = "_" + "_".join(features_to_add)

    final_name = f"{base_name}{appended_other_features_str}"

    return final_name, details

# --- Other Helpers ---

def get_job_result_safely(job_id, parent_task_id, task_type="child task"):
    """Safely retrieves the result of an RQ job, checking both RQ and the database.

    Always returns a dict with the same shape as the batch function's return value
    (contains 'status', 'best_result_from_batch', 'iterations_completed_in_batch',
    'final_subset_track_ids') so the caller can use a single code path, or None on failure.
    """
    # Local imports to prevent circular dependency
    from flask_app import app

    try:
        job = Job.fetch(job_id, connection=redis_conn)
        if job.is_finished and isinstance(job.result, dict):
            return job.result
    except NoSuchJobError:
        logger.warning(f"[{parent_task_id}] Job {job_id} not in RQ. Checking DB.")
        with app.app_context():
            task_info = get_task_info_from_db(job_id)
            if task_info and task_info.get('status') in [TASK_STATUS_SUCCESS, JobStatus.FINISHED]:
                try:
                    details = json.loads(task_info.get('details'))
                    # The DB stores the batch's internal details dict (with keys like
                    # 'full_best_result_from_batch', 'iterations_completed_in_batch', etc.)
                    # which is a DIFFERENT shape from the batch function's return value.
                    # Wrap it in the same envelope so the caller doesn't need special handling.
                    batch_result = details.get('full_best_result_from_batch') or details.get('full_result')
                    if batch_result:
                        return {
                            "status": "SUCCESS",
                            "best_result_from_batch": batch_result,
                            "iterations_completed_in_batch": details.get("iterations_completed_in_batch", 0),
                            "final_subset_track_ids": details.get("final_subset_track_ids", [])
                        }
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Could not parse result from DB for job {job_id}")
    return None

def _fill_balanced_quotas(quotas, capacities, remaining):
    remaining = max(0, int(remaining))
    while remaining > 0:
        open_genres = [
            genre for genre, capacity in capacities.items()
            if quotas.get(genre, 0) < capacity
        ]
        if not open_genres:
            break

        lowest_count = min(quotas.get(genre, 0) for genre in open_genres)
        lowest_genres = [
            genre for genre in open_genres
            if quotas.get(genre, 0) == lowest_count
        ]
        random.shuffle(lowest_genres)
        for genre in lowest_genres[:remaining]:
            quotas[genre] = quotas.get(genre, 0) + 1
            remaining -= 1
            if remaining == 0:
                break
    return quotas


def _calculate_stratified_quotas(genre_tracks, sample_size, target_per_genre):
    capacities = {
        genre: len(tracks)
        for genre, tracks in genre_tracks.items()
        if genre in STRATIFIED_GENRES and tracks
    }
    total_known = sum(capacities.values())
    wanted_known = min(max(0, int(sample_size)), total_known)
    target = max(0, int(target_per_genre))

    base_limits = {
        genre: min(capacity, target)
        for genre, capacity in capacities.items()
    }
    if sum(base_limits.values()) >= wanted_known:
        quotas = dict.fromkeys(capacities, 0)
        return _fill_balanced_quotas(quotas, base_limits, wanted_known)

    quotas = dict(base_limits)
    return _fill_balanced_quotas(
        quotas,
        capacities,
        wanted_known - sum(quotas.values()),
    )


def _regroup_tracks_by_primary_genre(genre_map):
    tracks_by_id = {}
    for tracks in genre_map.values():
        for track in tracks:
            track_id = track.get('item_id')
            if track_id is not None and track_id not in tracks_by_id:
                tracks_by_id[track_id] = track

    genre_tracks = defaultdict(list)
    for track in tracks_by_id.values():
        genre_tracks[_get_track_primary_genre(track)].append(track)
    return tracks_by_id, genre_tracks


def _select_tracks_for_genre(
    candidates, quota, previous_ids, change_fraction, selected_ids, rotate
):
    previous_candidates = [
        track for track in candidates if track['item_id'] in previous_ids
    ]
    keep_count = 0
    if rotate:
        keep_count = min(
            len(previous_candidates),
            quota,
            int(quota * (1.0 - change_fraction)),
        )
    kept = (
        random.sample(previous_candidates, keep_count)
        if keep_count < len(previous_candidates)
        else previous_candidates
    )
    chosen = list(kept)
    selected_ids.update(track['item_id'] for track in kept)

    needed = quota - len(kept)
    if needed <= 0:
        return chosen

    fresh = [
        track for track in candidates
        if track['item_id'] not in selected_ids
        and (change_fraction <= 0.0 or track['item_id'] not in previous_ids)
    ]
    added = random.sample(fresh, min(needed, len(fresh)))
    chosen.extend(added)
    selected_ids.update(track['item_id'] for track in added)

    still_needed = quota - len(kept) - len(added)
    if still_needed > 0:
        remaining = [
            track for track in candidates
            if track['item_id'] not in selected_ids
        ]
        reused = random.sample(remaining, still_needed)
        chosen.extend(reused)
        selected_ids.update(track['item_id'] for track in reused)
    return chosen


def _get_stratified_song_subset(
    genre_map,
    target_per_genre,
    prev_ids=None,
    percent_change=0.0,
):
    tracks_by_id, genre_tracks = _regroup_tracks_by_primary_genre(genre_map)

    desired_size = min(max(0, int(CLUSTERING_SUBSET_SONGS)), len(tracks_by_id))
    if desired_size == 0:
        return []

    quotas = _calculate_stratified_quotas(
        genre_tracks,
        desired_size,
        target_per_genre,
    )

    known_quota_total = sum(quotas.values())
    if known_quota_total < desired_size:
        other_capacity = len(genre_tracks.get('__other__', []))
        quotas['__other__'] = min(
            other_capacity,
            desired_size - known_quota_total,
        )

    previous_ids = set(prev_ids or [])
    change_fraction = min(1.0, max(0.0, float(percent_change)))
    rotate = prev_ids is not None
    selected, selected_ids = [], set()

    for genre, quota in quotas.items():
        if quota <= 0:
            continue
        selected.extend(
            _select_tracks_for_genre(
                genre_tracks.get(genre, []),
                quota,
                previous_ids,
                change_fraction,
                selected_ids,
                rotate,
            )
        )

    random.shuffle(selected)
    return selected

def _get_track_primary_genre(track_data):
    """Helper to determine the primary stratified genre for a track."""
    if 'mood_vector' in track_data and track_data['mood_vector']:
        mood_scores = {p.split(':')[0]: float(p.split(':')[1]) for p in track_data['mood_vector'].split(',') if ':' in p}
        return max((g for g in STRATIFIED_GENRES if g in mood_scores), key=mood_scores.get, default='__other__')
    return '__other__'

# Post-processing functions have been moved to clustering_postprocessing.py for better organization
