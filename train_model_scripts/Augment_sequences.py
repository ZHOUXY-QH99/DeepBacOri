#!/usr/bin/env python3
"""
Augment positive and negative samples by introducing random mutations.
This script reads FASTA files and generates new sequences with random mutations
to increase the sample size for model training.
"""

import os
import argparse
import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
from tqdm import tqdm

def mutate_sequence(sequence, mutation_rate=0.01, indel_rate=0.001, indel_length_range=(1, 5)):
    """
    Introduce random mutations to a sequence.
    
    Args:
        sequence: The sequence to mutate
        mutation_rate: Probability of point mutation (default: 0.01)
        indel_rate: Probability of insertion/deletion (default: 0.001)
        indel_length_range: Range of indel lengths (default: (1, 5))
    
    Returns:
        The mutated sequence
    """
    # Convert to list for easier manipulation
    seq_list = list(sequence)
    seq_len = len(seq_list)
    
    # Standard nucleotides
    standard_nucleotides = ['A', 'C', 'G', 'T']
    
    # Point mutations
    for i in range(seq_len):
        if random.random() < mutation_rate:
            # Choose a different nucleotide
            current = seq_list[i]
            if current in standard_nucleotides:
                # For standard nucleotides, choose a different one
                available_nucleotides = standard_nucleotides.copy()
                available_nucleotides.remove(current)
                seq_list[i] = random.choice(available_nucleotides)
            else:
                # For non-standard nucleotides, replace with a random standard one
                seq_list[i] = random.choice(standard_nucleotides)
    
    # Indels (insertions and deletions)
    i = 0
    while i < len(seq_list):
        if random.random() < indel_rate:
            # Decide if insertion or deletion
            if random.random() < 0.5:
                # Insertion
                indel_length = random.randint(indel_length_range[0], indel_length_range[1])
                for _ in range(indel_length):
                    seq_list.insert(i, random.choice(standard_nucleotides))
                i += indel_length
            else:
                # Deletion
                indel_length = random.randint(indel_length_range[0], min(indel_length_range[1], len(seq_list) - i))
                del seq_list[i:i+indel_length]
        
        i += 1
    
    # Convert back to string
    return ''.join(seq_list)

def augment_dataset(input_file, output_file, num_augmentations=5, mutation_rate=0.01, indel_rate=0.001):
    """
    Augment a dataset by generating new sequences with random mutations.
    
    Args:
        input_file: Path to input FASTA file
        output_file: Path to output FASTA file
        num_augmentations: Number of augmented sequences to generate per original sequence
        mutation_rate: Probability of point mutation
        indel_rate: Probability of insertion/deletion
    """
    print(f"Reading sequences from {input_file}...")
    records = list(SeqIO.parse(input_file, "fasta"))
    print(f"Found {len(records)} sequences")
    
    augmented_records = []
    
    # Process each sequence
    for i, record in enumerate(tqdm(records, desc="Augmenting sequences")):
        # Add original sequence
        augmented_records.append(record)
        
        # Generate augmented sequences
        for j in range(num_augmentations):
            # Create a mutated sequence
            mutated_seq = mutate_sequence(str(record.seq), mutation_rate, indel_rate)
            
            # Create a new record with a unique ID
            new_id = f"{record.id}_aug_{j+1}"
            new_record = SeqRecord(Seq(mutated_seq), id=new_id, description=f"Augmented from {record.id}")
            
            augmented_records.append(new_record)
    
    # Write augmented sequences to output file
    print(f"Writing {len(augmented_records)} sequences to {output_file}...")
    SeqIO.write(augmented_records, output_file, "fasta")
    print(f"Augmentation complete. Original: {len(records)}, Augmented: {len(augmented_records)}")

def main():
    parser = argparse.ArgumentParser(description='Augment positive and negative samples with random mutations')
    parser.add_argument('--positive', required=True, help='Path to positive samples FASTA file')
    parser.add_argument('--negative', required=True, help='Path to negative samples FASTA file')
    parser.add_argument('--output-dir', required=True, help='Directory to save augmented datasets')
    parser.add_argument('--num-augmentations', type=int, default=5, 
                      help='Number of augmented sequences to generate per original sequence (default: 5)')
    parser.add_argument('--mutation-rate', type=float, default=0.01,
                      help='Probability of point mutation (default: 0.01)')
    parser.add_argument('--indel-rate', type=float, default=0.001,
                      help='Probability of insertion/deletion (default: 0.001)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Augment positive samples
    positive_output = os.path.join(args.output_dir, "augmented_positive.fasta")
    augment_dataset(args.positive, positive_output, args.num_augmentations, 
                   args.mutation_rate, args.indel_rate)
    
    # Augment negative samples
    negative_output = os.path.join(args.output_dir, "augmented_negative.fasta")
    augment_dataset(args.negative, negative_output, args.num_augmentations, 
                   args.mutation_rate, args.indel_rate)
    
    # Combine augmented datasets
    combined_output = os.path.join(args.output_dir, "augmented_combined.fasta")
    print(f"Combining augmented datasets into {combined_output}...")
    
    combined_records = []
    combined_records.extend(SeqIO.parse(positive_output, "fasta"))
    combined_records.extend(SeqIO.parse(negative_output, "fasta"))
    
    # Shuffle the combined dataset
    random.shuffle(combined_records)
    
    # Write combined dataset
    SeqIO.write(combined_records, combined_output, "fasta")
    print(f"Combined dataset created with {len(combined_records)} sequences")
    
    # Print statistics
    print("\nAugmentation Statistics:")
    print(f"Original positive samples: {len(list(SeqIO.parse(args.positive, 'fasta')))}")
    print(f"Augmented positive samples: {len(list(SeqIO.parse(positive_output, 'fasta')))}")
    print(f"Original negative samples: {len(list(SeqIO.parse(args.negative, 'fasta')))}")
    print(f"Augmented negative samples: {len(list(SeqIO.parse(negative_output, 'fasta')))}")
    print(f"Total combined samples: {len(combined_records)}")

if __name__ == "__main__":
    main() 
