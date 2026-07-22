"""
config.py

Centralized configuration for the PV Power Forecasting project.
All experiment settings should be modified here instead of being
hardcoded throughout the project.
"""

import os
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle")


# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"


# =============================================================================
# Active Forecast Horizon
# =============================================================================
# Selects which preprocessed artifact pair main.py loads: "15" or "60",
# corresponding to TRAIN_15_FILE/VAL_15_FILE or TRAIN_60_FILE/VAL_60_FILE.
#
# This is also the single source of truth for the horizon segment used
# in every experiment/evaluation directory below. IS_KAGGLE has no
# bearing on this and must never be used to construct experiment paths;
# IS_KAGGLE is reserved exclusively for dataset paths (see the Dataset
# section further below).

ACTIVE_HORIZON = "15"


# =============================================================================
# Model Configuration
# =============================================================================

MODEL_NAME = "proposed"

# Directory hierarchy is Model -> Forecast Horizon -> Artifacts, so that
# artifacts from different forecast horizons (e.g. 15-minute vs.
# 60-minute training) never overwrite one another. The horizon segment
# is derived exclusively from ACTIVE_HORIZON above.
HORIZON_DIR_NAME = f"horizon_{ACTIVE_HORIZON}"

MODEL_EXPERIMENT_DIR = EXPERIMENTS_DIR / MODEL_NAME / HORIZON_DIR_NAME

MODEL_EVALUATION_DIR = EVALUATION_DIR / MODEL_NAME / HORIZON_DIR_NAME


# =============================================================================
# Multi-Run Experiment Management
# =============================================================================
# NUM_RUNS controls how many repeated runs are averaged for a given
# model/horizon combination (e.g. for reporting mean +/- std across
# seeds). RUN_NUMBER selects which individual run is currently active;
# it must be in [1, NUM_RUNS]. RUN_NAME derives the per-run directory
# segment from RUN_NUMBER and must never be set independently.
#
# Directory hierarchy becomes:
#   experiments/<model>/horizon_<h>/run_<n>/       (per-run artifacts)
#   experiments/<model>/horizon_<h>/average/       (aggregated across runs)
#   evaluation/<model>/horizon_<h>/run_<n>/
#   evaluation/<model>/horizon_<h>/average/

NUM_RUNS = 5
RUN_NUMBER = 1
RUN_NAME = f"run_{RUN_NUMBER}"

RUN_EXPERIMENT_DIR = MODEL_EXPERIMENT_DIR / RUN_NAME
RUN_EVALUATION_DIR = MODEL_EVALUATION_DIR / RUN_NAME

AVERAGE_EXPERIMENT_DIR = MODEL_EXPERIMENT_DIR / "average"
AVERAGE_EVALUATION_DIR = MODEL_EVALUATION_DIR / "average"

# Full list of per-run directories across all NUM_RUNS, used by
# aggregation utilities (e.g. computing mean/std across runs) so that
# run-path construction never needs to be duplicated outside config.py.
ALL_RUN_EXPERIMENT_DIRS = [
    MODEL_EXPERIMENT_DIR / f"run_{i}"
    for i in range(1, NUM_RUNS + 1)
]

ALL_RUN_EVALUATION_DIRS = [
    MODEL_EVALUATION_DIR / f"run_{i}"
    for i in range(1, NUM_RUNS + 1)
]


# =============================================================================
# Dataset
# =============================================================================

if IS_KAGGLE:
    RAW_DATA_FILE = Path(
    "/kaggle/input/datasets/aloksilswal/combined-output-all-arrays-csv/Combined_Output_All_Arrays.csv"
)

    # Processed dataset generated inside the notebook
    PROCESSED_DATA_FILE = Path("/kaggle/working/Processed.csv")

else:
    RAW_DATA_FILE = RAW_DATA_DIR / "Combined_Output_All_Arrays.csv"
    PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "Processed.csv"


# =============================================================================
# Processed Dataset Artifacts
# =============================================================================

TRAIN_15_FILE = ARTIFACT_DIR / "train_15.pt"
VAL_15_FILE = ARTIFACT_DIR / "val_15.pt"
TEST_15_FILE = ARTIFACT_DIR / "test_15.pt"

TRAIN_60_FILE = ARTIFACT_DIR / "train_60.pt"
VAL_60_FILE = ARTIFACT_DIR / "val_60.pt"
TEST_60_FILE = ARTIFACT_DIR / "test_60.pt"

FEATURE_SCALER_FILE = ARTIFACT_DIR / "feature_scaler.pkl"
TARGET_SCALER_FILE = ARTIFACT_DIR / "target_scaler.pkl"

PREPROCESSING_CONFIG_FILE = ARTIFACT_DIR / "preprocessing_config.json"


# =============================================================================
# Dataset Parameters
# =============================================================================

FEATURE_COLUMNS = [
    "Weather_Temperature_Celsius",
    "Weather_Relative_Humidity",
    "Global_Horizontal_Radiation",
    "Diffuse_Horizontal_Radiation",
    "Radiation_Global_Tilted",
    "Radiation_Diffuse_Tilted",
    "Active_Power",
]

TARGET_COLUMN = "Active_Power"

NUM_FEATURES = len(FEATURE_COLUMNS)


# =============================================================================
# Time-Series Parameters
# =============================================================================

LOOKBACK = 24

FORECAST_HORIZONS = [3, 12]      # 15 min and 60 min

STRIDE = 1


# =============================================================================
# Data Split
# =============================================================================

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# =============================================================================
# Training Parameters
# =============================================================================

BATCH_SIZE = 256

NUM_EPOCHS = 100

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

EARLY_STOPPING_PATIENCE = 15

# =============================================================================
# HPO Training Parameters
# =============================================================================
# Dedicated, reduced training settings used only while evaluating
# candidate configurations during hyperparameter optimization (see
# hpo/objective.py). Kept separate from NUM_EPOCHS and
# EARLY_STOPPING_PATIENCE above, which continue to govern final model
# training and are unaffected by these settings.

HPO_NUM_EPOCHS = 20

HPO_EARLY_STOPPING_PATIENCE = 5

# =============================================================================
# Reproducibility
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# DCNN Architecture
# =============================================================================

DCNN_NUM_CONV_LAYERS = 2          # Fixed

DCNN_FILTERS = 64                 # Default (HPO may change)
DCNN_KERNEL_SIZE = 3      # Fixed architecture default (not tuned by HPO)
DCNN_DILATION_RATE = 2     # Fixed architecture default (not tuned by HPO)
DCNN_DROPOUT_RATE = 0.20          # Default (HPO may change)

DCNN_STRIDE = 1                   # Fixed
DCNN_ACTIVATION = "relu"          # Fixed
DCNN_BATCH_NORM = True            # Fixed
DCNN_WEIGHT_INIT = "kaiming"      # Fixed

# =============================================================================
# Residual BiLSTM Architecture
# =============================================================================

BILSTM_NUM_LAYERS = 1             # Fixed

BILSTM_HIDDEN_SIZE = 64           # Default (HPO may change)
BILSTM_DROPOUT_RATE = 0.20        # Default (HPO may change)

BILSTM_BIDIRECTIONAL = True       # Fixed
BILSTM_BATCH_FIRST = True         # Fixed
BILSTM_RESIDUAL = True            # Fixed

BILSTM_WEIGHT_INIT = "custom"     # Xavier + Orthogonal + Forget Bias = 1

# =============================================================================
# Feature Attention
# =============================================================================

FEATURE_ATTENTION_REDUCTION = 8


# =============================================================================
# MLP Head
# =============================================================================

MLP_HIDDEN_DIM = 64        # Fixed architecture default (not tuned by HPO)
MLP_DROPOUT_RATE = 0.20    # Fixed architecture default (not tuned by HPO)


# =============================================================================
# Forecast Output Dimension
# =============================================================================

HORIZON_TO_OUTPUT_DIM = {
    "15": 3,
    "60": 12,
}


# =============================================================================
# DataLoader Parameters
# =============================================================================

SHUFFLE_TRAIN = True

NUM_WORKERS = 0


# =============================================================================
# Checkpointing
# =============================================================================

CHECKPOINT_DIR = RUN_EXPERIMENT_DIR / "checkpoints"

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best_checkpoint.pt"


# =============================================================================
# Gradient Clipping
# =============================================================================
# Set to None to disable gradient clipping.

GRADIENT_CLIP_VALUE = 1.0


# =============================================================================
# LR Scheduler (ReduceLROnPlateau)
# =============================================================================

SCHEDULER_MODE = "min"
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR = 1e-6


# =============================================================================
# Training History Output
# =============================================================================

HISTORY_FILE = RUN_EXPERIMENT_DIR / "history.json"

# =============================================================================
# Evaluation
# =============================================================================

MAX_PLOT_SAMPLES = 1000

EVALUATION_RESULTS_DIR = RUN_EVALUATION_DIR / "results"

EVALUATION_PLOTS_DIR = RUN_EVALUATION_DIR / "plots"

PREDICTIONS_FILE = EVALUATION_RESULTS_DIR / "predictions.csv"

EVALUATION_METRICS_FILE = EVALUATION_RESULTS_DIR / "evaluation_metrics.json"