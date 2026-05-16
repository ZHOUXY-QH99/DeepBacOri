"""
s01_data_preprocessing.py
Process FASTA files and create train/val/test splits (7:2:1 ratio)
"""
import os
import random
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split

# Configuration
DATA_DIR = "00_augmented_data"
SAVE_DIR = "01_processed"
POS_FILE = os.path.join(DATA_DIR, "augmented_positive.fasta")
NEG_FILE = os.path.join(DATA_DIR, "augmented_negative.fasta")
SEQ_LENGTH = 2000  # Truncate/pad sequences to this length

def process_sequences(pos_file, neg_file, seq_length):
    """Load and process FASTA files"""
    # Load positive samples
    pos_sequences = [str(rec.seq) for rec in SeqIO.parse(pos_file, "fasta")]
    pos_labels = [1] * len(pos_sequences)
    
    # Load negative samples
    neg_sequences = [str(rec.seq) for rec in SeqIO.parse(neg_file, "fasta")]
    neg_labels = [0] * len(neg_sequences)
    
    # Combine and shuffle
    all_sequences = pos_sequences + neg_sequences
    all_labels = pos_labels + neg_labels
    combined = list(zip(all_sequences, all_labels))
    random.shuffle(combined)
    sequences, labels = zip(*combined)
    
    # Convert to numerical format
    X = np.zeros((len(sequences), seq_length, 4), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j in range(min(seq_length, len(seq))):
            nucleotide = seq[j].upper()
            if nucleotide == 'A':
                X[i, j, 0] = 1
            elif nucleotide == 'T':
                X[i, j, 1] = 1
            elif nucleotide == 'C':
                X[i, j, 2] = 1
            elif nucleotide == 'G':
                X[i, j, 3] = 1
    
    y = np.array(labels, dtype=np.int64)
    return X, y

def save_splits(X, y, save_dir):
    """Split data and save numpy arrays"""
    # Split 70% train, 20% val, 10% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.333, stratify=y_temp)
    
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "y_train.npy"), y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "y_val.npy"), y_val)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "y_test.npy"), y_test)

if __name__ == "__main__":
    X, y = process_sequences(POS_FILE, NEG_FILE, SEQ_LENGTH)
    save_splits(X, y, SAVE_DIR)
    print(f"Data processing complete. Files saved to {SAVE_DIR}")
