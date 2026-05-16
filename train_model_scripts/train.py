"""
s02_model_training.py (Cleaned Version)
Train deep learning model on balanced dataset with clear SCI-style plots
"""
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# Configuration
DATA_DIR = "01_processed"
MODEL_DIR = "02_model"
BATCH_SIZE =128
EPOCHS = 30
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5  # Early stopping patience

class DNAModel(nn.Module):
    def __init__(self, seq_length=2000, n_features=4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=9)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=9)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.35)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.global_pool(x).squeeze(-1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X).squeeze()
            loss = criterion(outputs, y)
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.55).float()
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_true, all_preds)
    f1 = f1_score(all_true, all_preds)
    avg_loss = total_loss / len(data_loader)
    return accuracy, f1, avg_loss

def train_model(train_loader=None, val_loader=None, model=None, device=None):
    if train_loader is None or val_loader is None:
        X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
        y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
        X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
        y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = DNAModel().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    patience_counter = 0
    history = {
        'train_acc': [], 'train_f1': [], 'train_loss': [],
        'val_acc': [], 'val_f1': [], 'val_loss': []
    }

    print("Starting training...")
    print("Epoch | Train Acc | Train F1 | Val Acc | Val F1 | Train Loss | Val Loss")
    print("-" * 70)

    for epoch in range(EPOCHS):
        model.train()
        train_preds, train_true = [], []
        total_train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.55).float()
            train_preds.extend(preds.cpu().numpy())
            train_true.extend(y_batch.cpu().numpy())

        train_acc = accuracy_score(train_true, train_preds)
        train_f1 = f1_score(train_true, train_preds)
        train_loss = total_train_loss / len(train_loader)
        model.eval()
        val_preds, val_true = [], []
        total_val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X).squeeze()
                loss = criterion(outputs, y)
                total_val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.55).float()
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y.cpu().numpy())
        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds)
        val_loss = total_val_loss / len(val_loader)

        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_loss'].append(val_loss)

        print(f"{epoch+1:5d} | {train_acc:.4f} | {train_f1:.4f} | {val_acc:.4f} | {val_f1:.4f} | {train_loss:.4f} | {val_loss:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            model_path = os.path.join(MODEL_DIR, f"best_model_epoch{epoch+1}_f1_{val_f1:.4f}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'history': history
            }, model_path)
            print(f"Saved new best model at epoch {epoch+1} with F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    return model, history

def load_best_model(model_path, model=None):
    if model is None:
        model = DNAModel()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['history']

def plot_training_curves(history, save_path):
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Times New Roman']
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.labelsize'] = 16
    mpl.rcParams['axes.titlesize'] = 16
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 18
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'
    mpl.rcParams['xtick.major.width'] = 1.2
    mpl.rcParams['ytick.major.width'] = 1.2

    epochs = range(1, len(history['train_acc']) + 1)
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    lw = 2.5
    color_train = '#0072B2'
    color_val = '#D55E00'

    # Accuracy
    axs[0].plot(epochs, history['train_acc'], label='Train', color=color_train, linewidth=lw)
    axs[0].plot(epochs, history['val_acc'], label='Validation', color=color_val, linewidth=lw, linestyle='--')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_ylim(0.97, 1)
    axs[0].legend(frameon=False, fontsize=13, loc='lower right')
    axs[0].set_title('Model Accuracy')
    axs[0].spines['top'].set_visible(False)
    axs[0].spines['right'].set_visible(False)

    # F1 Score
    axs[1].plot(epochs, history['train_f1'], label='Train', color=color_train, linewidth=lw)
    axs[1].plot(epochs, history['val_f1'], label='Validation', color=color_val, linewidth=lw, linestyle='--')
    axs[1].set_ylabel('F1 Score')
    axs[1].set_ylim(0.97, 1)
    axs[1].legend(frameon=False, fontsize=13, loc='lower right')
    axs[1].set_title('Model F1 Score')
    axs[1].spines['top'].set_visible(False)
    axs[1].spines['right'].set_visible(False)

    # Loss
    axs[2].plot(epochs, history['train_loss'], label='Train', color=color_train, linewidth=lw)
    axs[2].plot(epochs, history['val_loss'], label='Validation', color=color_val, linewidth=lw, linestyle='--')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Loss')
    axs[2].set_ylim(0, 0.05)
    axs[2].legend(frameon=False, fontsize=13, loc='upper right')
    axs[2].set_title('Model Loss')
    axs[2].spines['top'].set_visible(False)
    axs[2].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    pdf_path = os.path.splitext(save_path)[0] + ".pdf"
    plt.savefig(pdf_path, format='pdf')
    plt.close()

def save_training_metadata(history, path):
    metadata = {
        'epoch': range(1, len(history['train_acc']) + 1),
        'train_acc': history['train_acc'],
        'train_f1': history['train_f1'],
        'train_loss': history['train_loss'],
        'val_acc': history['val_acc'],
        'val_f1': history['val_f1'],
        'val_loss': history['val_loss']
    }
    df = pd.DataFrame(metadata)
    df.to_csv(path, index=False)

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    model, history = train_model()
    print("Training complete. Best model saved in", MODEL_DIR)

    save_training_metadata(history, os.path.join(MODEL_DIR, "run_metadata.csv"))
    plot_training_curves(history, os.path.join(MODEL_DIR, "training_curves.png"))


