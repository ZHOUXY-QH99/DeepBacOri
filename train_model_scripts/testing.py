"""
s03_model_testing.py
Test the trained model performance on test set and prepare for subsequent predictions
"""
import os
import torch
import numpy as np
from Bio import SeqIO
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Configuration parameters (must match training settings)
DATA_DIR = "01_processed"
MODEL_DIR = "02_model"  # Load model from 02_model
OUTPUT_DIR = "03_test"  # Save results to 03_output
BATCH_SIZE = 64
SEQ_LENGTH = 2000

class DNAModel(torch.nn.Module):
    """Model structure must be identical to training code"""
    def __init__(self, seq_length=2000, n_features=4):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(n_features, 64, kernel_size=9)
        self.pool = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=9)
        self.global_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.fc1 = torch.nn.Linear(128, 64)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(64, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def preprocess_sequence(seq):
    """Preprocess a single DNA sequence
    - Convert sequence to uppercase
    - Handle N bases as zero vectors (no information)
    - Encode ATCG as one-hot vectors
    """
    seq = seq.upper()
    encoded = np.zeros((SEQ_LENGTH, 4), dtype=np.float32)
    for i, nt in enumerate(seq[:SEQ_LENGTH]):
        if nt == 'A': encoded[i, 0] = 1
        elif nt == 'T': encoded[i, 1] = 1
        elif nt == 'C': encoded[i, 2] = 1
        elif nt == 'G': encoded[i, 3] = 1
        # N bases remain as zero vectors
    return encoded

def load_model():
    """Load the best trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DNAModel().to(device)
    
    # Load model from 02_model
    model_path = os.path.join(MODEL_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load complete checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    # Only load model state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    return model, device

def test_model():
    """Main testing process"""
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Load model
        model, device = load_model()
        
        # Load test data
        X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
        y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
        
        # Create test DataLoader
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32)
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Perform predictions
        all_preds = []
        all_probs = []
        all_true = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch).squeeze()
                
                # Get prediction probabilities
                probs = torch.sigmoid(outputs).cpu().numpy()
                # Convert to class predictions
                preds = (probs > 0.5).astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_true.extend(y_batch.numpy())
        
        # Calculate evaluation metrics
        print("\n=== Test Results ===")
        print(f"Accuracy: {accuracy_score(all_true, all_preds):.4f}")
        print(f"F1 Score: {f1_score(all_true, all_preds):.4f}")
        print(f"ROC AUC: {roc_auc_score(all_true, all_probs):.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_true, all_preds))
        print("\nClassification Report:")
        print(classification_report(all_true, all_preds))
        
        # Save prediction results to CSV file
        results = np.column_stack((all_probs, all_preds, all_true))
        results_path = os.path.join(OUTPUT_DIR, "test_results.csv")
        np.savetxt(
            results_path,
            results,
            delimiter=",",
            header="Probability,Prediction,TrueLabel",
            comments=""
        )
        print(f"\nPrediction results saved to: {results_path}")
        
        # Copy trained model to 03_output directory for subsequent predictions
        import shutil
        shutil.copy2(
            os.path.join(MODEL_DIR, "best_model.pth"),
            os.path.join(OUTPUT_DIR, "best_model.pth")
        )
        print(f"\nModel copied to: {os.path.join(OUTPUT_DIR, 'best_model.pth')}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        print("Please ensure model training is completed and model file exists in correct location.")
        exit(1)
    except Exception as e:
        print(f"Error during testing process: {str(e)}")
        exit(1)

if __name__ == "__main__":
    test_model()

