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
# Aggressive + shorter range on GRMN6694_360 (stride2, N=902):
#   y50 + max_lanes 3 + |x|<=5.5  vs  mild y100/L4/x7:
#   avgL 2.85→2.45, >=4 33%→0%, CONF 71.0%→73.4%, corr ~91% held.
CONF_THRESHOLD = 0.43          # was 0.40; mild raise drops weak extras
NMS_THRES_M = 2.0              # was 3.0; mean lateral distance (meters)
MAX_LANES = 3                  # was 4/6; ego L/R + one adjacent only
MIN_VISIBLE_POINTS = 4         # was effectively 2; drop short/noisy segments
MIN_ABS_MEAN_X_M = 0.0         # >0 rejects near-center ghosts but hurts ego pair
MAX_ABS_MEAN_X_M = 5.5         # was 7/8; cut shoulder / far laterals
MAX_LATERAL_JUMP_M = 2.5       # max |Δx| between adjacent visible Y samples
MAX_ABS_SLOPE = 0.35           # max |Δx / Δy| over visible span (filters diagonals)
# Only use / plot / track anchors with Y <= this (meters). Model still predicts to 100.
MAX_LANE_Y_M = 50.0            # was 100; far anchors add noise more than signal

# --- Synthetic lane interpolation (draw path) ---
ENABLE_FILL_MISSING_LANES = False  # was always on; invents fake lines
FILL_MIN_GAP_M = 6.0               # only used if ENABLE_FILL_MISSING_LANES=True
FILL_MIN_SCORE = 0.50              # both neighbors must be >= this score if scores exist

# Shared temporal hold: EKF coast, ego-pair, and onesided reconstruct
# must use the same budget so tracks and the drivable fill drop together.
LANE_HOLD_FRAMES = 15          # ~0.5s at 30fps

# --- Temporal EKF tracking ---
EKF_MAX_MISSED_FRAMES = LANE_HOLD_FRAMES
EKF_DIST_THRESHOLD_M = 1.8     # association distance threshold
EKF_CONFIRM_HITS = 2           # show lane after 2 consecutive detections
EKF_REQUIRE_CONFIRMED = False  # immediately render active tracks without long confirmation delay

# --- Ego corridor / P0 lane-pair (ADAS) ---
EGO_LANE_WIDTH_MIN_M = 2.8     # reject pairs narrower than a real lane
# Model-space lane width has centimetre-level frame variation; leave a small
# acceptance margin above the nominal 4.6 m limit without admitting 2 lanes.
EGO_LANE_WIDTH_MAX_M = 4.8     # reject pairs that span 2+ lanes
EGO_LANE_WIDTH_TARGET_M = 3.7  # prefer pairs near standard lane width
EGO_CORRIDOR_MARGIN_M = 0.24   # inset so fill sits inside ego paint, not on adjacent
CORRIDOR_WIDTH_MAX_M = 3.9     # if ego pair is wider, shrink to target around center
CORRIDOR_WIDTH_CLAMP_M = 3.7   # width used when clamping an oversized pair
# Always draw the fill at this paint-to-paint width (then inset by margin),
# including EKF / onesided PREDICTED fallback — stops the corridor from ballooning.
CORRIDOR_FORCE_FIXED_WIDTH = True
# Front fill starts past the ego hood (image bottom). Same idea as YOLO_BOTTOM_DROP_FRAC.
CORRIDOR_IMAGE_HOOD_FRAC = 0.14
CORRIDOR_Y_START_M = 4.0       # also skip 3D samples closer than this (hood / bumper)
EGO_PAIR_HOLD_FRAMES = LANE_HOLD_FRAMES
EGO_PAIR_MATCH_X_M = 1.25      # rematch held lanes to new proposals by |Δmean_x|
# Occupancy-first ego pair: kill adjacent-lane latch.
# A pair "occupies" ego when each boundary is at least INNER meters from X=0.
# Center weight dominates; width is only a tie-break. Do not use |center|<1.2
# as a hard cap — ADAS clips often have ~1.5 m camera-frame bias.
EGO_OCCUPANCY_INNER_M = 0.40
EGO_CENTER_SCORE_W = 3.0
EGO_WIDTH_SCORE_W = 0.25
EGO_REQUIRE_CONTAINS_0 = True  # never pick a pair that does not contain X=0
# If no occupying pair exists, allow a weaker contains-0 pair (left<0<right)
# whose |center| is still below FALLBACK_MAX_CENTER (neighbour latch is ~1.6–1.9 m).
EGO_OCCUPANCY_FALLBACK_CONTAINS0 = True
EGO_FALLBACK_MAX_CENTER_M = 1.59
# closest-left + closest-right re-locks the adjacent lane; keep off.
EGO_LEGACY_FALLBACK = False

# --- One-sided ego reconstruct (P1) ---
# If only one ego paint line is measured, rebuild the missing side from a
# locked width. Visualization / BEV may use it; CIPO stays fail-closed
# because RoadState.status will be PREDICTED, not CONFIRMED.
ENABLE_ONESIDED_RECONSTRUCT = True
ONESIDED_MAX_Y_M = 40.0        # only invent the missing side in the near field
ONESIDED_HOLD_FRAMES = LANE_HOLD_FRAMES
ONESIDED_MIN_LOCK_FRAMES = 3   # confirmed pairs required before trusting W
ONESIDED_W_EMA_ALPHA = 0.20    # slow width lock (higher = follow new gaps more)
ONESIDED_MATCH_X_M = 1.50      # rematch the live side to last ego X

# --- Corridor temporal EMA (P2) ---
CORRIDOR_EMA_ALPHA = 0.35      # higher = trust new frame more
CORRIDOR_EMA_MAX_JUMP_M = 1.8  # reject / hard-switch if lateral jump exceeds this

# --- P3 dark / low-light (CLAHE + adaptive conf) ---
# Garmin A/B (GRMN6694_540_nohud, N=1803):
#   CLAHE (gated or always) *hurts* corridor — OpenLane engine domain shift.
#   Lower conf on dark frames only *helps* (+46 corridor, both-gone 43%→64%).
ENABLE_DARK_CLAHE = False
ENABLE_CLAHE_ALWAYS = False
ENABLE_ADAPTIVE_CONF = True
DARK_LUMA_MAX = 125.0           # mean HSV-V of lower 2/3 of the model image
CLAHE_CLIP = 2.5
CLAHE_TILE = 8
DARK_CONF_THRESHOLD = 0.38      # slightly below day conf for dark frames

# --- CIPO / P1 in-path hysteresis ---
CIPO_ENTER_HITS = 2            # frames inside before marking in_path
CIPO_EXIT_MISS = 4             # frames outside before clearing in_path
CIPO_U_MARGIN_PX = 3.0         # pixel slack around projected ego lines
CIPO_X_MARGIN_M = 0.35         # extra meters beyond measured ego half-width
