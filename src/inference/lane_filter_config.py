"""
Lane robustness quick-win knobs.

Tune these, restart the app, and compare lane clutter vs miss rate.
Start with the defaults below, then adjust one knob at a time.

Suggested sweeps:
  CONF_THRESHOLD : 0.35 → 0.45 → 0.55   (higher = fewer lanes)
  NMS_THRES_M    : 2.5  → 2.0  → 1.5    (lower  = stronger de-dupe)
  MAX_LANES      : 6 → 4                (hard cap after NMS)
  ENABLE_FILL_MISSING_LANES : False unless ego corridor looks incomplete
"""

# --- Detection scoring / NMS (postprocess) ---
CONF_THRESHOLD = 0.40          # was 0.20; raise if still too many FPs
NMS_THRES_M = 2.0              # was 3.0; mean lateral distance (meters)
MAX_LANES = 6                  # keep top-K by score after NMS (0 = unlimited)
MIN_VISIBLE_POINTS = 4         # was effectively 2; drop short/noisy segments
MAX_ABS_MEAN_X_M = 8.0         # drop lanes farther than this from ego centerline
MAX_LATERAL_JUMP_M = 2.5       # max |Δx| between adjacent visible Y samples
MAX_ABS_SLOPE = 0.35           # max |Δx / Δy| over visible span (filters diagonals)

# --- Synthetic lane interpolation (draw path) ---
ENABLE_FILL_MISSING_LANES = False  # was always on; invents fake lines
FILL_MIN_GAP_M = 6.0               # only used if ENABLE_FILL_MISSING_LANES=True
FILL_MIN_SCORE = 0.50              # both neighbors must be >= this score if scores exist

# --- Temporal EKF tracking ---
EKF_MAX_MISSED_FRAMES = 5      # was 10; drop ghost tracks sooner
EKF_DIST_THRESHOLD_M = 1.8     # was 2.5; tighter association
EKF_CONFIRM_HITS = 5           # was 2–3; only draw after this many matches
EKF_REQUIRE_CONFIRMED = True   # if True, never draw unconfirmed tracks
