#Evaluate_model.py 

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import config # Import the centralized configuration
# Import model definition and dataset class from train_model.py or separate file

# Assuming necessary functions (clean_text, tokenize, load_and_preprocess_data)
# and classes (BiGRUClassifier, TransactionDataset) are accessible/imported

def evaluate_model():
    # 1. Load Data (assuming load_and_preprocess_data returns X_train, X_temp, etc.)
    # In a real setup, you'd load a dedicated test set here.
    df = pd.read_csv(config.DATA_PATH)
    # ... (similar preprocessing steps as in train_model.py to get X_test, y_test)
    
    # Simulate loading test data:
    X_test = ["starbucks nyc", "amazon returns", "shell gas station"] # Example test set
    y_test_labels = ["Coffee/Dining", "Shopping", "Fuel"]
    
    # 2. Load Artifacts and Model
    vocab = joblib.load(config.VOCAB_SAVE_PATH)
    label_encoder = joblib.load(config.ENCODER_SAVE_PATH)
    NUM_CLASSES = len(label_encoder.classes_)
    y_test = label_encoder.transform(y_test_labels) 

    # 3. Create Test DataLoader
    test_dataset = TransactionDataset(pd.Series(X_test), pd.Series(y_test), vocab)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    # 4. Initialize and Load Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiGRUClassifier(
        vocab_size=len(vocab),
        embed_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        num_classes=NUM_CLASSES
    ).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 5. Run Evaluation
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. Generate Report
    print("\n--- Classification Report ---")
    print(classification_report(
        all_labels, 
        all_preds, 
        target_names=label_encoder.classes_,
        # Use macro average for F1-Score as specified in your proposal
        average='macro' 
    )) 

if __name__ == '__main__':
    evaluate_model()
