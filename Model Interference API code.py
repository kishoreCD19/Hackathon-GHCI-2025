# 2. api_main.py

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import joblib
import os
import re
import string
import numpy as np
from typing import List, Dict

# Configuration
# Must match the configuration in train_model.py
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2 
MAX_LEN = 20 # Max sequence length for padding

# 1. Load Preprocessing Artifacts and Model 
try:
    LABEL_ENCODER = joblib.load('label_encoder.joblib')
    VOCAB = joblib.load('vocab.joblib')
    NUM_CLASSES = len(LABEL_ENCODER.classes_)
except FileNotFoundError:
    print("FATAL ERROR: Model artifacts not found. Run 'train_model.py' first.")
    exit(1)

# 2. Model Definition (Must be identical to the training script) 
class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        _, hidden = self.gru(embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        return self.fc(hidden)

# 3. Preload Model 
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = BiGRUClassifier(
    vocab_size=len(VOCAB),
    embed_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES
).to(DEVICE)

# Load the trained weights
MODEL.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
MODEL.eval()

# 4. Preprocessing for Inference 
# Use the same functions as in the training script
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) 
    text = ' '.join(text.split())
    return text

def tokenize(text):
    return text.split()

def preprocess_for_inference(raw_description: str) -> torch.Tensor:
    """Cleans, tokenizes, numericalizes, and pads the input text."""
    cleaned_text = clean_text(raw_description)
    
    # Tokenize and numericalize
    token_ids = [VOCAB[token] for token in tokenize(cleaned_text)]
    
    # Padding/Truncation
    padded_tokens = token_ids[:MAX_LEN] + [VOCAB['<pad>']] * (MAX_LEN - len(token_ids))
    
    # Convert to PyTorch tensor (batch size 1)
    return torch.tensor(padded_tokens).unsqueeze(0).to(DEVICE)


# 5. FastAPI App and Models 
app = FastAPI(
    title="AutoClassify AI", 
    description="Real-time AI Transaction Categorisation Service."
)

class TransactionIn(BaseModel):
    """Input model for the classification request."""
    raw_description: str
    amount: float = None # Not used in this simple model, but good for feature engineering

class ClassificationOut(BaseModel):
    """Output model for the classification response."""
    raw_description: str
    predicted_category: str
    confidence_score: float

# 6. API Endpoint 

@app.post("/classify", response_model=ClassificationOut)
async def classify_transaction(transaction: TransactionIn):
    """
    Classifies a single transaction description using the trained Bi-GRU model.
    """
    
    # 1. Preprocess input
    input_tensor = preprocess_for_inference(transaction.raw_description)
    
    # 2. Run Inference
    with torch.no_grad():
        output = MODEL(input_tensor)
        
    # 3. Get Prediction and Confidence
    probabilities = F.softmax(output, dim=1)
    confidence, predicted_id = torch.max(probabilities, 1)
    
    # Convert to standard Python types
    predicted_label = LABEL_ENCODER.inverse_transform(predicted_id.cpu().numpy())[0]
    confidence_score = confidence.item()

    return ClassificationOut(
        raw_description=transaction.raw_description,
        predicted_category=predicted_label,
        confidence_score=round(confidence_score, 4)
    )

@app.get("/health")
async def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "ok", "model_loaded": True}
