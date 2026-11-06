"""Fixed experiment constants (mirrors preregistration.json)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
TABLES = ROOT / "paper" / "tables"

SEEDS = (0, 1, 2, 3, 4)
ALPHA = 0.05
Q_FDR = 0.05

# Detector settings
N_BINS = 10
SMOOTH = 0.5
N_MIN = 30  # windows smaller than this abstain (result = unknown)
PSI_RULES = (0.1, 0.25)
PSI_BIN_ABLATION = (5, 10, 20)

# Detector benchmark on real features
REF_SIZE = 1000
WINDOW_SIZES = (100, 500, 2000)
N_NULL_CAL = 200
N_NULL_EVAL = 200
N_DRIFT = 100
FPR_TARGET = 0.05
MAGNITUDES = (0.05, 0.1, 0.2, 0.4)
NUMERIC_DRIFTS = ("mean_shift", "scale_change", "selection_shift")
CATEGORICAL_DRIFTS = ("frequency_shift",)

# Streaming delay benchmark
STREAM_N = 200
STREAM_PRE = 10
STREAM_POST = 30
STREAM_DELTA_MAX = 0.5
STREAM_REPS = 20
STREAM_NULL_CAL = 500
STREAM_FPR = 0.01

# Portfolio simulation
PORTFOLIO_SIZES = (10, 100, 1000)
N_FEATURES = 20
PORTFOLIO_REF = 1000
PORTFOLIO_WINDOW = 200
N_DAYS = 30
DRIFT_MODEL_FRACTION = 0.1
DRIFT_FEATURES_MAX = 5
DELTA_LO, DELTA_HI = 0.1, 0.5
ONSET_LO, ONSET_HI = 5, 20
RECALL_HORIZON = 3
CORR_RHO = 0.6
MISSING_RATE = 0.1
ABLATION_WINDOWS = (100, 500)

# Electricity agreement study
WEEK = 336  # 48 half-hour periods x 7 days
TRAIN_WEEKS = 8
VAL_WEEKS = 2
MONITOR_WEEKS = 40
TRAIN_START_WEEKS = (52, 60, 68, 76, 84)
RF_TREES = 100
MIN_LABELS = 30
KAPPA_BOUND = 0.4

# Timing
TIMING_SIZES = (1000, 10000)
TIMING_REPS = 200
