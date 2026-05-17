# DeepBacOri: Deep Learning-based Bacterial Replication Origin Predictor

DeepBacOri is a comprehensive tool designed to predict DNA replication origin sites (*oriC*) in bacterial genomes using a deep learning approach integrated with multi-evidence bioinformatics analysis. It provides robust predictions for both complete genomes and fragmented Metagenome-Assembled Genomes (MAGs).


## 🛠 Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Biopython
- NumPy, SciPy
- Matplotlib

### Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/DeepBacOri.git
cd DeepBacOri
pip install torch biopython numpy scipy matplotlib
```

## 📖 Usage

### 1. Complete Genome Prediction
Use `prediction_complete_genome.py` for high-quality finished genomes. It supports both single file analysis and batch processing of directories.

#### Single File Mode
```bash
python prediction_complete_genome.py -i sequence.fasta -g annotation.gbk -o output_dir
```

#### Directory (Batch) Mode
```bash
python prediction_complete_genome.py -d input_fasta_dir -gd gbk_dir -o output_dir
```

**Parameters:**
- `-i, --input`: Input FASTA file (single genome).
- `-d, --input_dir`: Directory containing multiple FASTA files.
- `-g, --gene_file`: GenBank (.gbk) file for single genome analysis.
- `-gd, --gene_dir`: Directory containing GenBank files for batch mode.
- `-o, --output`: Output directory for results.
- `--indicator_genes`: Comma-separated list of genes to search for (default: `dnaA,dnaN,mnmG`).

### 2. MAG Prediction
Use `prediction_MAG.py` for metagenomic contigs or MAGs. Similar to the complete genome script, it supports both single file and directory inputs.

#### Single File Mode
```bash
python prediction_MAG.py -i mag_sequence.fasta -g mag_annotation.gbk -o output_dir
```

#### Directory (Batch) Mode
```bash
python prediction_MAG.py -d input_mags_dir -gd gbk_dir -o output_dir
```

**Parameters:**
- `-i, --input`: Input FASTA file (single MAG).
- `-d, --input_dir`: Directory containing multiple FASTA files for batch prediction.
- `-g, --gene_file`: GenBank/GBFF gene annotation file for single MAG.
- `-gd, --gene_dir`: Directory containing GenBank/GBFF gene files for batch mode.
- `-o, --output`: Output directory for all results.
- `--batch_size_mags`: Number of MAGs to process in each batch (default: 10).
- `--max_workers`: Maximum number of worker processes for parallel processing (default: 2).

## 📊 Output Files

The tool generates an output directory containing:
- **`*_report.html`**: A comprehensive visual report containing:
  - Global prediction probability plots.
  - Circular/Linear GC skew visualizations.
  - Z-curve and DnaA box density plots.
  - Flanking gene structure diagrams.
- **`*_predictions.csv`**: Tabular data of predicted regions with coordinates and probabilities.
- **`*_flanking10kb_genes.csv`**: List of genes within 10kb of the predicted *oriC*.
- **`plots/`**: Individual PNG files for all generated visualizations.

## 🧬 Methodology

DeepBacOri identifies *oriC* through a hierarchical scoring system:
1. **Initial Screening**: A 1D-CNN slides through the genome to identify high-probability origin regions.
2. **Structural Validation**: Integration of Z-curve extrema and GC skew minima.
3. **Motif Search**: Clusters of DnaA boxes (TTATNCACA) are identified and scored.
4. **Genomic Context**: Proximity to indicator genes (like *dnaA*, *dnaN*) is evaluated.
5. **Recommendation**: A final score is calculated to recommend the most likely *oriC* candidate.



