import os
import torch
from dotenv import load_dotenv

load_dotenv()

DATASET_PATH = os.getenv("DATASET_PATH")
DATA = os.path.join(DATASET_PATH, "new-potato-leaf-diseases-dataset")
TEST_SIZE = 0.3
VALI_SIZE = 0.5
RANDOM_STATE = 42
BATCH_SIZE = 128
LEARNING_RATE = 2e-5
EPOCHS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
