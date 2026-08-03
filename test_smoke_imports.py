#!/usr/bin/env python3
"""Structural import smoke test — no DB, no ONNX, no Flask context needed."""
import sys, os, traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

errors = []
successes = []

def try_import(label, statement):
    try:
        exec(statement, {})
        successes.append(f"  OK {label}")
    except Exception as e:
        tb = traceback.format_exc()
        errors.append(f"  FAIL {label}: {e}")
        if len(errors) <= 2:
            print(tb[:500])

# 1. Config
try_import("config vars",
    "import config; "
    "c = config.CLUSTERING_MODE; "
    "a = config.ANALYSIS_MODE; "
    "h = config.HYBRID_PCA_MUSICNN; "
    "l = config.LN_HYBRID_MOOD_DIVERSITY_STATS; "
    "i = config.INDEX_NAME_MAEST; "
    "ac = config.AUTO_CALIBRATE_LN_STATS; "
    "fw = config.FUSION_WEIGHT_MUSICNN_DEFAULT"
)

# 2. clustering_fusion
try_import("clustering_fusion", "from tasks import clustering_fusion")

# 3. ivf_manager (just imports, no DB init)
try_import("ivf_manager",
    "import tasks.ivf_manager as m; "
    "f = m._both_indexes_loaded; "
    "b = m.build_and_store_maest_ivf_index; "
    "fn = m.find_nearest_neighbors_fused"
)

# 4. clustering_helper
try_import("clustering_helper",
    "import tasks.clustering_helper as h; "
    "c = h._calibrate_ln_stats; "
    "r = h._resolve_ln_stats; "
    "hs = h.HybridScaler"
)

# 5. commons
try_import("commons", "from tasks.commons import score_vector")

# 6. clustering_postprocessing
try_import("clustering_postprocessing", "import tasks.clustering_postprocessing as cp")

# 7. analysis.song (the MAEST functions)
try_import("analysis.song MAEST fns",
    "from tasks.analysis.song import ("
    "analyze_track_maest, prepare_maest_melspectrogram, "
    "run_maest_inference, load_maest_session, "
    "cleanup_maest_session, persist_maest_results)"
)

# 8. analysis.helper (TrackPlan, planning)
try_import("analysis.helper TrackPlan",
    "from tasks.analysis.helper import ("
    "TrackPlan, plan_track_stages, "
    "get_existing_maest_track_ids, build_album_plan); "
    "tp = TrackPlan(musicnn=True, maest=False, clap=False, lyrics=False); "
    "a = tp.any_stage; d = tp.describe(); n = tp.needs_audio"
)

# 9. app_clustering (Flask blueprint — may fail without Flask)
try_import("app_clustering", "import app_clustering")

# 10. app_ivf
try_import("app_ivf", "import app_ivf")

# 11. database — save_maest_embedding + dynamic params
try_import("database dynamic params",
    "from database import save_maest_embedding, save_track_analysis_and_embedding; "
    "import inspect; "
    "sig = inspect.signature(save_track_analysis_and_embedding); "
    "assert 'mood_column' in sig.parameters, 'missing mood_column param'; "
    "assert 'embedding_table' in sig.parameters, 'missing embedding_table param'"
)

# 12. app_analysis
try_import("app_analysis", "import app_analysis")

# 13. app_dashboard
try_import("app_dashboard", "import app_dashboard")

# 14. app_external
try_import("app_external", "import app_external")

# 15. app.py (the big one — Flask app with routes)
try_import("app", "import app")

print(f"\n{'='*50}")
print(f"Results: {len(successes)} passed, {len(errors)} failed")
for e in errors:
    print(e)