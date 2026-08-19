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

# --- Ego corridor / P0 lane-pair (ADAS) ---
EGO_LANE_WIDTH_MIN_M = 2.8     # reject pairs narrower than a real lane
# Model-space lane width has centimetre-level frame variation; leave a small
# acceptance margin above the nominal 4.6 m limit without admitting 2 lanes.
EGO_LANE_WIDTH_MAX_M = 4.8     # reject pairs that span 2+ lanes
EGO_LANE_WIDTH_TARGET_M = 3.7  # prefer pairs near standard lane width
EGO_CORRIDOR_MARGIN_M = 0.12   # light inset so red fill matches ego lane width in BEV/front
EGO_PAIR_HOLD_FRAMES = 8       # keep last good pair during lane-change gaps
EGO_PAIR_MATCH_X_M = 1.25      # rematch held lanes to new proposals by |Δmean_x|

# --- Corridor temporal EMA (P2) ---
CORRIDOR_EMA_ALPHA = 0.35      # higher = trust new frame more
CORRIDOR_EMA_MAX_JUMP_M = 1.8  # reject / hard-switch if lateral jump exceeds this

# --- CIPO / P1 in-path hysteresis ---
CIPO_ENTER_HITS = 2            # frames inside before marking in_path
CIPO_EXIT_MISS = 4             # frames outside before clearing in_path
CIPO_U_MARGIN_PX = 3.0         # pixel slack around projected ego lines
CIPO_X_MARGIN_M = 0.35         # extra meters beyond measured ego half-width
