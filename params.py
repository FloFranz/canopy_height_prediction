"""
Data and train parameters.
"""
import torch

DIR_MODELS = "models/"
DIR_METRICS = "metrics/"
DIR_DATA = "E:/Datasets/FORTRESS/data/datasets/"              # Change this to your dataset root directory
DIR_DATA_ORTHOMOSAICS = "orthomosaic/"
DIR_DATA_HEIGHTMAPS = "nDSM/"

FILENAME = "{DIR}{NAME}{EXT}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXCLUSION_LIST = []

PROGRESS_BAR_TRAIN = 16
PROGRESS_BAR_VALIDATE = 6

# Hyperparameters
HP_TRAIN_FRACTION = 0.7
HP_TEST_FRACTION = 0.1
HP_EPOCHS = 5
HP_BATCH_SIZE = 20
HP_LEARNING_RATE = 1e-4
HP_WEIGHT_DECAY = 1e-5