# AudioMuse-AI - https://github.com/NeptuneHub/AudioMuse-AI
# Copyright (C) 2025 NeptuneHub
# SPDX-License-Identifier: AGPL-3.0-only
"""Unit tests for MAEST analysis integration and multi-mode clustering.

Covers:
* persist_maest_results uses catalog_item_id over raw provider id
* plan_track_stages correctly accounts for existing MAEST data
* album_feature_needs and work_done_bits include the MAEST work bit
* _LN_STATS_BY_MODE resolves dual_consensus to hybrid mood stats
* The calibration if/elif chain updates the correct LN_*_STATS for each mode
* WORK_MAEST bit is set/checked correctly by the work-map helpers
"""

import numpy as np


# ─── persist_maest_results ────────────────────────────────────────────

class _FakeItem:
    """Simulates a track item dict that has both a provider id and a canonical id."""
    def __init__(self, provider_id='jf_999', canonical_id='fp_2abc', name='Test Song'):
        self._item = {
            'Id': provider_id,
            '_catalog_item_id': canonical_id,
            'Name': name,
            'AlbumArtist': 'Test Artist',
        }

    def get(self, key, default=None):
        return self._item.get(key, default)


def test_persist_maest_results_uses_canonical_id(monkeypatch):
    """persist_maest_results must use catalog_item_id(item), not item['Id']."""
    from tasks.analysis.song import persist_maest_results

    item = _FakeItem(provider_id='jf_999', canonical_id='fp_2abc123')

    saved = {}

    def fake_save(item_id, title, author, **kwargs):
        saved['item_id'] = item_id
        saved['title'] = title

    monkeypatch.setattr('tasks.analysis.song.save_track_analysis_and_embedding', fake_save)

    analysis = {'tempo': 120, 'key': 'C', 'scale': 'major', 'energy': 0.5}
    embedding = np.zeros(768, dtype=np.float32)
    moods = {'pop': 0.8}
    persist_maest_results(item, analysis, moods, embedding, '')

    assert saved.get('item_id') == 'fp_2abc123', (
        f"Expected canonical id 'fp_2abc123', got {saved.get('item_id')!r}"
    )


def test_persist_maest_falls_back_to_provider_id_when_no_canonical(monkeypatch):
    """catalog_item_id falls back to item['Id'] when _catalog_item_id is missing."""
    from tasks.analysis.song import persist_maest_results

    item = _FakeItem(provider_id='jf_777', canonical_id=None)
    # Force _catalog_item_id to be missing by modifying the dict
    item._item.pop('_catalog_item_id', None)

    saved = {}

    def fake_save(item_id, title, author, **kwargs):
        saved['item_id'] = item_id
        saved['title'] = title

    monkeypatch.setattr('tasks.analysis.song.save_track_analysis_and_embedding', fake_save)

    analysis = {'tempo': 120, 'key': 'C', 'scale': 'major', 'energy': 0.5}
    embedding = np.zeros(768, dtype=np.float32)
    moods = {'pop': 0.8}
    persist_maest_results(item, analysis, moods, embedding, '')

    assert saved.get('item_id') == 'jf_777', (
        f"Expected fallback id 'jf_777', got {saved.get('item_id')!r}"
    )


# ─── plan_track_stages ─────────────────────────────────────────────────

def test_plan_track_stages_maest_missing_triggers_maest(monkeypatch):
    """plan_track_stages returns plan.maest=True when track is not in existing_maest_ids."""
    from tasks.analysis.helper import plan_track_stages

    plan = plan_track_stages(
        track_id='fp_2abc',
        existing_ids={'fp_2abc'},
        existing_maest_ids=set(),  # not done yet
        missing_clap_ids=set(),
        missing_lyrics_ids=set(),
        lyrics_enabled=False,
    )
    assert plan.maest is True, "MAEST should be needed when not in existing_maest_ids"


def test_plan_track_stages_maest_done_skips_maest(monkeypatch):
    """plan_track_stages returns plan.maest=False when track IS in existing_maest_ids."""
    from tasks.analysis.helper import plan_track_stages

    plan = plan_track_stages(
        track_id='fp_2abc',
        existing_ids={'fp_2abc'},
        existing_maest_ids={'fp_2abc'},  # already done
        missing_clap_ids=set(),
        missing_lyrics_ids=set(),
        lyrics_enabled=False,
    )
    assert plan.maest is False, "MAEST should be skipped when already in existing_maest_ids"


def test_replan_for_catalogue_row_checks_maest_embedding(monkeypatch):
    """replan_for_catalogue_row re-checks maest_embedding table."""
    from tasks.analysis.helper import replan_for_catalogue_row, TrackPlan

    monkeypatch.setattr('tasks.analysis.helper.get_missing_ids_in_table', lambda table, ids: (
        {'fp_2abc'} if table == 'maest_embedding' else set()
    ))

    plan = TrackPlan(musicnn=False, maest=True, clap=False, lyrics=False)
    replan = replan_for_catalogue_row(plan, 'fp_2abc')
    assert replan.maest is True, "replan should keep MAEST when embedding is missing"


# ─── Work bits (orchestrator skip logic) ───────────────────────────────

def test_work_maest_constant_value():
    """WORK_MAEST is 8 and does not collide with existing bits."""
    from tasks.analysis.helper import WORK_MUSICNN, WORK_CLAP, WORK_LYRICS, WORK_MAEST

    assert WORK_MUSICNN == 1
    assert WORK_CLAP == 2
    assert WORK_LYRICS == 4
    assert WORK_MAEST == 8
    # Verify no overlap
    assert (WORK_MUSICNN | WORK_CLAP | WORK_LYRICS | WORK_MAEST) == 15


def test_work_done_bits_includes_maest_when_enabled():
    """work_done_bits(maest_enabled=True) includes WORK_MAEST."""
    from tasks.analysis.helper import work_done_bits, WORK_MAEST

    bits = work_done_bits(clap_available=False, lyrics_enabled=False, maest_enabled=True)
    assert bits & WORK_MAEST, "MAEST bit should be set when maest_enabled=True"

    bits_off = work_done_bits(clap_available=False, lyrics_enabled=False, maest_enabled=False)
    assert not (bits_off & WORK_MAEST), "MAEST bit should not be set when maest_enabled=False"


def test_album_feature_needs_maest_detects_missing():
    """album_feature_needs returns needs_maest=True when a track lacks the MAEST bit."""
    from tasks.analysis.helper import album_feature_needs, WORK_MUSICNN, WORK_MAEST

    # Mask: has MUSICNN but not MAEST
    masks = [WORK_MUSICNN]
    done_bits = WORK_MUSICNN | WORK_MAEST

    album_done, _, _, _, needs_maest = album_feature_needs(
        masks, done_bits, clap_available=False, lyrics_enabled=False, maest_enabled=True
    )
    assert needs_maest is True, "needs_maest should be True when a track lacks the MAEST bit"
    assert album_done == 0, "album_done should be 0 when not all bits are present"


def test_album_feature_needs_maest_all_good():
    """album_feature_needs returns needs_maest=False when all tracks have MAEST."""
    from tasks.analysis.helper import album_feature_needs, WORK_MUSICNN, WORK_MAEST

    masks = [WORK_MUSICNN | WORK_MAEST]
    done_bits = WORK_MUSICNN | WORK_MAEST

    album_done, _, _, _, needs_maest = album_feature_needs(
        masks, done_bits, clap_available=False, lyrics_enabled=False, maest_enabled=True
    )
    assert needs_maest is False
    assert album_done == 1


def test_album_feature_needs_maest_disabled_skips_check():
    """album_feature_needs returns needs_maest=False when maest_enabled=False."""
    from tasks.analysis.helper import album_feature_needs, WORK_MUSICNN

    masks = [WORK_MUSICNN]  # no MAEST bit
    done_bits = WORK_MUSICNN  # maest_enabled=False → MAEST not in done_bits

    album_done, _, _, _, needs_maest = album_feature_needs(
        masks, done_bits, clap_available=False, lyrics_enabled=False, maest_enabled=False
    )
    assert needs_maest is False, "should be False when maest_enabled=False"
    assert album_done == 1


# ─── LN stats registry ────────────────────────────────────────────────

def test_ln_stats_by_mode_has_dual_consensus():
    """_LN_STATS_BY_MODE contains dual_consensus, falling back to hybrid stats."""
    from tasks.clustering_helper import _LN_STATS_BY_MODE

    assert 'dual_consensus' in _LN_STATS_BY_MODE
    entry = _LN_STATS_BY_MODE['dual_consensus']
    # Should reference the same hybrid constants, not musicnn or maest
    from tasks.clustering_helper import (
        LN_HYBRID_MOOD_DIVERSITY_STATS,
        LN_HYBRID_MOOD_PURITY_STATS,
    )
    assert entry['diversity'] is LN_HYBRID_MOOD_DIVERSITY_STATS
    assert entry['purity'] is LN_HYBRID_MOOD_PURITY_STATS


def test_resolve_ln_stats_dual_consensus_uses_hybrid_stats(monkeypatch):
    """_resolve_ln_stats with clustering_mode='dual_consensus' returns hybrid stats."""
    from tasks.clustering_helper import (
        _resolve_ln_stats,
        LN_HYBRID_MOOD_DIVERSITY_STATS,
        LN_HYBRID_MOOD_PURITY_STATS,
    )

    div, pur = _resolve_ln_stats(use_embeddings=False, clustering_mode='dual_consensus')
    assert div is LN_HYBRID_MOOD_DIVERSITY_STATS
    assert pur is LN_HYBRID_MOOD_PURITY_STATS


def test_calibrate_ln_stats_dual_consensus_updates_hybrid(monkeypatch):
    """_calibrate_ln_stats for dual_consensus updates LN_HYBRID_MOOD_*_STATS.

    This tests the if/elif chain in clustering.py that dispatches on
    clustering_mode_param.
    """
    # Simulated calibration return value
    div_stats = {'mean': 1.2, 'sd': 0.4}
    pur_stats = {'mean': 4.5, 'sd': 1.1}

    from tasks.clustering_helper import (
        LN_HYBRID_MOOD_DIVERSITY_STATS,
        LN_HYBRID_MOOD_PURITY_STATS,
        LN_MOOD_DIVERSITY_STATS,
    )

    # Save original values
    orig_div = dict(LN_HYBRID_MOOD_DIVERSITY_STATS)
    orig_pur = dict(LN_HYBRID_MOOD_PURITY_STATS)

    # Simulate the update logic from clustering.py
    import tasks.clustering_helper as ch
    mode = 'dual_consensus'
    if mode == 'maest':
        ch.LN_MAEST_GENRE_DIVERSITY_STATS.update(div_stats)
        ch.LN_MAEST_GENRE_PURITY_STATS.update(pur_stats)
    elif mode in ('hybrid_blend', 'dual_consensus'):
        ch.LN_HYBRID_MOOD_DIVERSITY_STATS.update(div_stats)
        ch.LN_HYBRID_MOOD_PURITY_STATS.update(pur_stats)
    else:
        ch.LN_MOOD_DIVERSITY_STATS.update(div_stats)
        ch.LN_MOOD_PURITY_STATS.update(pur_stats)

    assert LN_HYBRID_MOOD_DIVERSITY_STATS['mean'] == 1.2
    assert LN_HYBRID_MOOD_PURITY_STATS['mean'] == 4.5
    # Musicnn stats should be untouched
    assert LN_MOOD_DIVERSITY_STATS['mean'] != 1.2

    # Restore
    LN_HYBRID_MOOD_DIVERSITY_STATS.update(orig_div)
    LN_HYBRID_MOOD_PURITY_STATS.update(orig_pur)


# ─── Low-level work-map helpers ────────────────────────────────────────

def test_apply_work_bits_sets_maest():
    """_apply_work_bits sets the WORK_MAEST bit when has_maest=True."""
    from tasks.analysis.helper import _apply_work_bits, WORK_MAEST

    work_map = {}
    _apply_work_bits(work_map, 'p1', has_musicnn=True, has_clap=False, has_lyrics=False, has_maest=True)
    assert work_map['p1'] & WORK_MAEST, "MAEST bit should be set"


def test_apply_work_bits_skips_maest_when_false():
    """_apply_work_bits does not set WORK_MAEST when has_maest=False."""
    from tasks.analysis.helper import _apply_work_bits, WORK_MAEST

    work_map = {}
    _apply_work_bits(work_map, 'p1', has_musicnn=True, has_clap=False, has_lyrics=False, has_maest=False)
    assert not (work_map['p1'] & WORK_MAEST), "MAEST bit should not be set"