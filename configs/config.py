import os
import torch

RAW_DIR = "data/raw/gdsc"
PROCESSED_DIR = "data/processed"
CHECKPOINT_DIR = "models/checkpoints"
RESULTS_DIR = "assets/results"

TOP_N_GENES = 1000
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

ATOM_FEATURE_DIM = 20
GENE_FEATURE_DIM = TOP_N_GENES
HIDDEN_DIM = 128
GAT_HEADS = 4
DROPOUT = 0.2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 32
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

HOLDOUT_CANCER_TYPES = [
    "Kidney Carcinoma",
    "Pancreatic Carcinoma",
    "Hepatocellular Carcinoma",
]

MC_DROPOUT_SAMPLES = 50
BOOTSTRAP_RESAMPLES = 1000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
