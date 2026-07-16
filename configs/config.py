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
# Model Configuration
# =============================================================================

MODEL_NAME = "proposed"

MODEL_EXPERIMENT_DIR = EXPERIMENTS_DIR / MODEL_NAME

MODEL_EVALUATION_DIR = EVALUATION_DIR / MODEL_NAME


# =============================================================================
# Dataset
# =============================================================================

if IS_KAGGLE:
    RAW_DATA_FILE = Path(
        "/kaggle/input/datasets/Combined_Output_All_Arrays.csv"
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

EARLY_STOPPING_PATIENCE = 10

# =============================================================================
# Reproducibility
# =============================================================================

RANDOM_SEED = 42

# =============================================================================
# DCNN Architecture
# =============================================================================

DCNN_NUM_CONV_LAYERS = 2          # Fixed

DCNN_FILTERS = 64                 # Default (HPO may change)
DCNN_KERNEL_SIZE = 3              # Default (HPO may change)
DCNN_DILATION_RATE = 2            # Default (HPO may change)
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

MLP_HIDDEN_DIM = 64          # Default (HPO may change)
MLP_DROPOUT_RATE = 0.20      # Default (HPO may change)


# =============================================================================
# Active Forecast Horizon
# =============================================================================
# Selects which preprocessed artifact pair main.py loads: "15" or "60",
# corresponding to TRAIN_15_FILE/VAL_15_FILE or TRAIN_60_FILE/VAL_60_FILE.

ACTIVE_HORIZON = "15"

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

CHECKPOINT_DIR = MODEL_EXPERIMENT_DIR / "checkpoints"

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

HISTORY_FILE = MODEL_EXPERIMENT_DIR / "history.json"

# =============================================================================
# Evaluation
# =============================================================================

MAX_PLOT_SAMPLES = 1000

EVALUATION_RESULTS_DIR = MODEL_EVALUATION_DIR / "results"

EVALUATION_PLOTS_DIR = MODEL_EVALUATION_DIR / "plots"

PREDICTIONS_FILE = EVALUATION_RESULTS_DIR / "predictions.csv"

EVALUATION_METRICS_FILE = EVALUATION_RESULTS_DIR / "evaluation_metrics.json"