#1. config.py 

import os

# Model Hyperparameters
MODEL_NAME = "BiGRU_Classifier_v1.2"
VOCAB_SIZE = 10000        # Max words for vocabulary
EMBEDDING_DIM = 128       # Size of word embedding vectors
HIDDEN_DIM = 256          # Size of GRU hidden state
NUM_LAYERS = 2            # Number of GRU layers
MAX_LEN = 30              # Max sequence length for padding
DROPOUT_RATE = 0.3        # Dropout rate for regularization

# Training Parameters 
NUM_EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
RANDOM_STATE = 42

# File Paths 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "transactions_labeled.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
VOCAB_SAVE_PATH = os.path.join(BASE_DIR, "models", "vocab.joblib")
ENCODER_SAVE_PATH = os.path.join(BASE_DIR, "models", "label_encoder.joblib")

# API Configuration 
API_HOST = "0.0.0.0"
API_PORT = 8000
API_KEY_HEADER = "X-API-Key" # Header name for security

#Thresholds
CONFIDENCE_THRESHOLD = 0.85 # Minimum score to accept prediction without flagging
