Data processing model - training python model.train_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchtext.vocab import build_vocab_from_iterator
import re
import string
import joblib

# Configuration
VOCAB_SIZE = 5000
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
NUM_EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001

#1.1 Preprocessing Functions
def clean_text(text):
    """Clean transaction text: remove punctuation, lowercase, strip whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Keep only alphanumeric and spaces
    text = ' '.join(text.split())
    return text

def tokenize(text):
    """Simple tokenization by splitting the cleaned string."""
    return text.split()

# 1.2 Data Loading 
def load_and_preprocess_data():
    """Simulate loading and preprocessing a transaction dataset."""
    # Create a dummy dataset (replace with your actual data loading)
    data = {
        'raw_description': [
            'STARBUCKS #1234 NYC', 'AMAZON.COM*1A2B C3D', 'SHELL OIL 445 GAS',
            'UBER TRIP 01FEB', 'CVS PHARMACY #998', 'WALMART SUPERCENTER',
            'STARBUCKS #5555 LA', 'AMZN MKTPLACE PMTS', 'CHEVRON GAS STATION',
            'UBEREATS DELIVERY', 'WALGREENS #101', 'WHOLE FOODS MARKET'
        ],
        'category_label': [
            'Coffee/Dining', 'Shopping', 'Fuel', 'Transport', 'Groceries/Pharmacy',
            'Groceries/Pharmacy', 'Coffee/Dining', 'Shopping', 'Fuel',
            'Transport', 'Groceries/Pharmacy', 'Groceries/Pharmacy'
        ]
    }
    df = pd.DataFrame(data)

    # 1. Clean Text
    df['cleaned_text'] = df['raw_description'].apply(clean_text)

    # 2. Encode Labels
    label_encoder = LabelEncoder()
    df['category_id'] = label_encoder.fit_transform(df['category_label'])
    NUM_CLASSES = len(label_encoder.classes_)

    print(f"Number of Categories: {NUM_CLASSES}")
    print(f"Category Map: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # 3. Create Vocabulary (FastText-like approach uses only words)
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenize(text)

    # Build vocabulary from cleaned text
    vocab = build_vocab_from_iterator(
        yield_tokens(df['cleaned_text']),
        max_tokens=VOCAB_SIZE,
        specials=["<unk>", "<pad>"]
    )
    vocab.set_default_index(vocab["<unk>"])
    
    # 4. Split Data
    X_train, X_temp, y_train, y_temp = train_test_split(
        df['cleaned_text'], df['category_id'], test_size=0.3, random_state=42, stratify=df['category_id']
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    return X_train, y_train, X_val, y_val, vocab, label_encoder, NUM_CLASSES

# 1.3 PyTorch Dataset and Model 

class TransactionDataset(Dataset):
    def __init__(self, X, y, vocab):
        self.X = X.tolist()
        self.y = y.tolist()
        self.vocab = vocab
        self.max_len = 20 # Max length for padding

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        text = self.X[idx]
        label = self.y[idx]
        
        # Tokenize and numericalize
        token_ids = [self.vocab[token] for token in tokenize(text)]
        
        # Padding
        padded_tokens = token_ids[:self.max_len] + [self.vocab['<pad>']] * (self.max_len - len(token_ids))
        
        return torch.tensor(padded_tokens), torch.tensor(label, dtype=torch.long)

class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True
        )
        # 2 * hidden_dim because it's bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded shape: [batch_size, seq_len, embed_dim]
        
        _, hidden = self.gru(embedded)
        # hidden shape: [num_layers * 2, batch_size, hidden_dim] (Bi-directional * 2)

        # Concatenate the last forward and backward layer
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden shape: [batch_size, hidden_dim * 2]
        
        output = self.fc(hidden)
        # output shape: [batch_size, num_classes]
        return output

# 1.4 Training Function 

def train_model(model, train_loader, val_loader, criterion, optimizer, device):
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # Training loop
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Model saved: best_model.pth")


if __name__ == '__main__':
    # 1. Load Data
    X_train, y_train, X_val, y_val, vocab, label_encoder, NUM_CLASSES = load_and_preprocess_data()

    # 2. Save Preprocessing Artifacts for Inference API
    joblib.dump(label_encoder, 'label_encoder.joblib')
    joblib.dump(vocab, 'vocab.joblib')
    print("Preprocessing artifacts saved.")

    # 3. Create DataLoaders
    train_dataset = TransactionDataset(X_train, y_train, vocab)
    val_dataset = TransactionDataset(X_val, y_val, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 4. Initialize Model, Loss, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiGRUClassifier(
        vocab_size=len(vocab),
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 5. Train Model
    print(f"Starting training on device: {device}")
    train_model(model, train_loader, val_loader, criterion, optimizer, device)
    print("Training complete.")
