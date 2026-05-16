#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
s06_prediction_MAG_fixed_verbose_optimized_batch.py
Predict replication origin sites in MAGs using trained model
Optimized version with batch processing to avoid CUDA OOM
"""

import os
import argparse
import gzip
import csv
import base64
import re
import time
import signal
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from Bio import SeqIO
from torch.utils.data import DataLoader, TensorDataset

# Global flag for graceful shutdown
_shutdown_flag = False

# Model path relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_SCRIPT_DIR, "model", "model.pth")
SEQ_LENGTH = 2000          # Fixed, model trained with this length
STEP_SIZE = 500            # Fixed sliding window step
BATCH_SIZE = 64            # Fixed batch size for DL prediction
THRESHOLD = 0.8
INDICATOR_GENES = ["dnaA", "dnaN", "mnmG"]

class DNAModel(torch.nn.Module):
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


class DeepLearningPredictor:
    def __init__(self, model_path=MODEL_PATH, use_gpu=True):
        self.device = torch.device("cpu")
        if use_gpu and torch.cuda.is_available():
            if torch.cuda.memory_allocated() < 2 * 1024 * 1024 * 1024:
                self.device = torch.device("cuda")
        self.model = self._load_model(model_path)
        self.seq_length = SEQ_LENGTH
        self.step_size = STEP_SIZE
        self.batch_size = BATCH_SIZE

    def _load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = DNAModel().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _encode_sequence(self, seq):
        seq = seq.upper().replace("N", "A")
        encoded = np.zeros((self.seq_length, 4), dtype=np.float32)
        mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
        for i, nt in enumerate(seq[: self.seq_length]):
            if nt in mapping:
                encoded[i, mapping[nt]] = 1.0
        return encoded

    def predict_sequence(self, seq):
        if len(seq) < self.seq_length:
            return [], []
        window_starts = list(range(0, len(seq) - self.seq_length + 1, self.step_size))
        if not window_starts:
            return [], []

        windows = []
        positions = []
        for start in window_starts:
            if start + self.seq_length <= len(seq):
                windows.append(self._encode_sequence(seq[start : start + self.seq_length]))
                positions.append(start)

        return self._batch_predict(windows, positions)

    def _batch_predict(self, windows, positions):
        if not windows:
            return [], []

        windows_array = np.array(windows)
        dataset = TensorDataset(torch.tensor(windows_array, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=min(self.batch_size * 2, len(windows_array)))

        probs = []
        with torch.no_grad():
            for batch in loader:
                inputs = batch[0].to(self.device)
                outputs = self.model(inputs).squeeze()
                batch_probs = torch.sigmoid(outputs).cpu().numpy()
                if batch_probs.ndim == 0:
                    probs.append(float(batch_probs.item()))
                else:
                    probs.extend([float(x) for x in batch_probs])

        return positions, probs

    def find_peaks(self, positions, probs, threshold=THRESHOLD):
        peaks = []
        peak_probs = []
        for i in range(len(positions)):
            if probs[i] > threshold:
                peaks.append(positions[i])
                peak_probs.append(probs[i])
        return peaks, peak_probs

    def plot_global_probability(self, positions, probs, seq_id, save_path):
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(positions, probs, linewidth=1, label="Prediction Probability", alpha=0.7)
        ax.set_xlabel("Genome Position", fontsize=12)
        ax.set_ylabel("Probability", fontsize=12)
        ax.set_title(f"Prediction Probability: {seq_id}", fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return save_path


class DnaABoxAnalyzer:
    @staticmethod
    def find_dnaa_boxes_optimized(sequence, patterns=None, max_mismatches=1):
        patterns = patterns or {"DnaA_box": "TTATNCACA"}
        boxes = []
        seq_upper = sequence.upper()
        seq_len = len(sequence)

        for pattern_name, pattern_seq in patterns.items():
            pattern_length = len(pattern_seq)
            pattern_upper = pattern_seq.upper()

            for i in range(seq_len - pattern_length + 1):
                segment = seq_upper[i:i + pattern_length]
                mismatches = 0
                for j in range(pattern_length):
                    if pattern_upper[j] == 'N':
                        continue
                    if segment[j] != pattern_upper[j]:
                        mismatches += 1
                        if mismatches > max_mismatches:
                            break
                if mismatches <= max_mismatches:
                    boxes.append({
                        "start": i,
                        "end": i + pattern_length,
                        "sequence": segment,
                        "pattern": pattern_name,
                        "mismatches": mismatches
                    })
        return sorted(boxes, key=lambda x: x["start"])

    def run_dnaa_box_analysis_optimized(self, seq, seq_id, output_dir):
        boxes = self.find_dnaa_boxes_optimized(seq)
        if not boxes:
            return boxes, None

        window_size = 1000
        seq_len = len(seq)
        num_windows = (seq_len + window_size - 1) // window_size
        positions = np.arange(window_size // 2, window_size * num_windows, window_size)
        counts = np.zeros(num_windows, dtype=int)

        for box in boxes:
            window_idx = box["start"] // window_size
            if window_idx < num_windows:
                counts[window_idx] += 1

        plot_path = os.path.join(output_dir, f"{seq_id}_dnaa_box.png")
        self._plot_dnaa_analysis_optimized(positions[:len(counts)], counts, plot_path)
        return boxes, plot_path

    def _plot_dnaa_analysis_optimized(self, positions, counts, plot_path):
        plt.figure(figsize=(12, 4))
        plt.plot(positions, counts, label="DnaA Box Count", linewidth=1, alpha=0.8)
        plt.title("DnaA Box Count per 1kb Window")
        plt.xlabel("Position")
        plt.ylabel("Number of DnaA Boxes")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()


class GeneStructureAnalyzer:
    def __init__(self, verbose=False, output_dir="."):
        self.verbose = verbose
        self.output_dir = output_dir
        self.log_file = None
        self._gbk_cache = {}

    def setup_log_file(self, base_name):
        log_filename = f"{base_name}_gene_matching.log"
        self.log_file = os.path.join(self.output_dir, log_filename)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def _log(self, message, to_console=False):
        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a") as f:
                f.write(f"{timestamp}: {message}\n")
        if to_console and self.verbose:
            print(f"[DEBUG] {message}")

    def _normalize_contig_id(self, contig_id):
        if not contig_id:
            return ""
        s = str(contig_id).lstrip(">").strip()
        token = s.split()[0]
        token = re.sub(r"(?:\.\d+)+$", "", token)
        token = re.sub(r"/\d+$", "", token)
        return token

    def _collect_record_ids(self, record):
        ids = set()
        try:
            if getattr(record, "id", None):
                ids.add(str(record.id))
        except Exception:
            pass
        try:
            if getattr(record, "name", None):
                ids.add(str(record.name))
        except Exception:
            pass
        try:
            if getattr(record, "description", None):
                ids.add(str(record.description))
        except Exception:
            pass
        norm = set(self._normalize_contig_id(x) for x in ids if x)
        return norm

    def load_gbk_cache(self, gbk_path):
        if gbk_path in self._gbk_cache:
            return self._gbk_cache[gbk_path]

        cache = {}
        if not gbk_path or not os.path.exists(gbk_path):
            return cache

        opener = gzip.open if gbk_path.endswith(".gz") else open
        try:
            with opener(gbk_path, "rt") as handle:
                for record in SeqIO.parse(handle, "genbank"):
                    gbk_ids = self._collect_record_ids(record)
                    genes = []
                    for feature in record.features:
                        if feature.type not in ("CDS", "gene"):
                            continue
                        try:
                            location = feature.location
                            start = int(location.start)
                            end = int(location.end)
                            strand = 1
                            if hasattr(location, 'strand') and location.strand is not None:
                                strand = int(location.strand)
                            qualifiers = getattr(feature, 'qualifiers', {})
                            locus_tag = qualifiers.get("locus_tag", [""])[0]
                            gene = qualifiers.get("gene", [""])[0]
                            product = qualifiers.get("product", [""])[0]
                            if not gene and locus_tag:
                                gene = locus_tag
                            genes.append({
                                "start": start,
                                "end": end,
                                "strand": strand,
                                "locus_tag": locus_tag,
                                "gene": gene,
                                "product": product,
                                "feature_type": feature.type
                            })
                        except Exception:
                            continue
                    for gbk_id in gbk_ids:
                        if gbk_id:
                            cache[gbk_id] = sorted(genes, key=lambda x: x["start"])
                    if hasattr(record, 'id') and record.id:
                        cache[self._normalize_contig_id(record.id)] = sorted(genes, key=lambda x: x["start"])
            self._gbk_cache[gbk_path] = cache
            self._log(f"Loaded GBK cache with {len(cache)} contig entries")
            return cache
        except Exception as e:
            self._log(f"Error loading GBK cache {gbk_path}: {str(e)}")
            return {}

    def get_genes_for_contig(self, gbk_path, target_contig_id):
        if not gbk_path:
            return []
        cache = self.load_gbk_cache(gbk_path)
        if not cache:
            return []

        target_variants = set()
        if target_contig_id:
            target_variants.add(str(target_contig_id))
            normalized = self._normalize_contig_id(target_contig_id)
            if normalized:
                target_variants.add(normalized)
            base_id = re.sub(r'(\.\d+)+$', '', str(target_contig_id))
            if base_id:
                target_variants.add(base_id)
            first_token = str(target_contig_id).split()[0] if target_contig_id else ""
            if first_token:
                target_variants.add(first_token)
                target_variants.add(self._normalize_contig_id(first_token))

        for target in target_variants:
            if target in cache:
                genes = cache[target]
                self._log(f"Found {len(genes)} genes for contig '{target_contig_id}' using variant '{target}'")
                return genes

        for cache_key in cache.keys():
            for target in target_variants:
                if target and cache_key and (target in cache_key or cache_key in target):
                    genes = cache[cache_key]
                    self._log(f"Found {len(genes)} genes for contig '{target_contig_id}' via substring match")
                    return genes
        self._log(f"No genes found for contig '{target_contig_id}'")
        return []

    def find_indicator_genes(self, genes, indicator_genes=None):
        if indicator_genes is None:
            indicator_genes = INDICATOR_GENES
        indicator_genes_lower = [g.lower() for g in indicator_genes]
        found = []
        for gene in genes:
            name = (gene.get("gene") or "").lower()
            prod = (gene.get("product") or "").lower()
            locus = (gene.get("locus_tag") or "").lower()
            for indicator in indicator_genes_lower:
                if indicator in name or indicator in prod or indicator in locus:
                    found.append({**{k: gene.get(k, "") for k in ["gene", "product", "locus_tag"]},
                                 "start": gene["start"], "end": gene["end"],
                                 "strand": gene.get("strand", 1), "indicator_gene": indicator})
                    break
        return found

    def plot_gene_structure_optimized(self, seq, genes, ori_start, ori_end, save_path):
        ori_center = (int(ori_start) + int(ori_end)) // 2
        window_start = max(0, ori_center - 10000)
        window_end = min(len(seq), ori_center + 10000)

        region_genes = []
        for g in genes:
            if g["end"] > window_start and g["start"] < window_end:
                region_genes.append(g)

        if not region_genes:
            fig, ax = plt.subplots(figsize=(10, 3), dpi=100)
            ax.set_xlim(window_start, window_end)
            ax.set_ylim(-1, 2)
            ax.set_xlabel("Genome Position", fontsize=10)
            ax.set_ylabel("Gene Layer", fontsize=10)
            ax.set_title("Gene Structure (10kb flanking OriC) - No genes found", fontsize=12)
            ax.axvspan(ori_start, ori_end, color="red", alpha=0.2, label="Predicted oriC")
            ax.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            return save_path

        region_genes_sorted = sorted(region_genes, key=lambda x: x["start"])
        y_levels = []
        gene_positions = []
        for gene in region_genes_sorted:
            assigned_level = 0
            while True:
                overlap = False
                for i, (g_start, g_end, level) in enumerate(gene_positions):
                    if level == assigned_level and not (gene["end"] < g_start or gene["start"] > g_end):
                        overlap = True
                        break
                if not overlap:
                    break
                assigned_level += 1
            gene_positions.append((gene["start"], gene["end"], assigned_level))
            y_levels.append(assigned_level)

        fig, ax = plt.subplots(figsize=(12, max(4, max(y_levels) * 0.5 + 3)), dpi=100)
        for gene, y_level in zip(region_genes_sorted, y_levels):
            gene_color = "blue" if gene.get("strand", 1) == 1 else "red"
            gene_length = gene["end"] - gene["start"]
            if gene.get("strand", 1) == 1:
                ax.arrow(gene["start"], y_level, gene_length, 0,
                        head_width=0.3, head_length=max(100, gene_length * 0.1),
                        fc=gene_color, ec=gene_color, length_includes_head=True,
                        linewidth=1, alpha=0.7)
            else:
                ax.arrow(gene["end"], y_level, -gene_length, 0,
                        head_width=0.3, head_length=max(100, gene_length * 0.1),
                        fc=gene_color, ec=gene_color, length_includes_head=True,
                        linewidth=1, alpha=0.7)
            label_text = gene.get("gene") or gene.get("locus_tag") or ""
            if label_text and len(label_text) < 20:
                label_x = (gene["start"] + gene["end"]) / 2
                ax.text(label_x, y_level + 0.1, label_text,
                       ha="center", va="bottom", fontsize=6, rotation=45)

        ax.axvspan(ori_start, ori_end, color="orange", alpha=0.3, label="Predicted oriC")
        ax.set_xlim(window_start - 500, window_end + 500)
        ax.set_ylim(-1, max(y_levels) + 2)
        ax.set_xlabel("Genome Position (bp)", fontsize=10)
        ax.set_ylabel("Gene Layer", fontsize=10)
        ax.set_title("Gene Structure (10kb upstream and downstream from OriC center)", fontsize=12)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="orange", alpha=0.3, label="Predicted oriC"),
            Patch(facecolor="blue", alpha=0.7, label="Forward strand"),
            Patch(facecolor="red", alpha=0.7, label="Reverse strand")
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        return save_path

    def get_flanking_genes(self, genes, ori_start, ori_end, flanking_distance=10000):
        center = (int(ori_start) + int(ori_end)) // 2
        window_start = max(0, center - flanking_distance)
        window_end = min(center + flanking_distance,
                        max((g["end"] for g in genes), default=center + flanking_distance))
        flanking = [g for g in genes if g["end"] >= window_start and g["start"] <= window_end]
        return sorted(flanking, key=lambda x: x["start"])


class ResultExporter:
    def write_flanking10kb_genes_csv(self, contig_id, regions, genes, output_csv):
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["contig_id", "region_rank", "region_start", "region_end", "region_center",
                           "flanking_window_start", "flanking_window_end", "gene_start", "gene_end",
                           "gene_name", "product", "strand", "gene_length", "distance_to_ori_center"])
            for i, region in enumerate(regions, 1):
                region_start = int(region["start"])
                region_end = int(region["end"])
                region_center = (region_start + region_end) // 2
                window_start = max(0, region_center - 10000)
                window_end = min(region_center + 10000, max((g["end"] for g in genes), default=region_center + 10000))
                for g in genes:
                    if g["end"] < window_start or g["start"] > window_end:
                        continue
                    gene_center = (g["start"] + g["end"]) // 2
                    distance = gene_center - region_center
                    writer.writerow([
                        contig_id,
                        i,
                        region_start,
                        region_end,
                        region_center,
                        window_start,
                        window_end,
                        g["start"],
                        g["end"],
                        g.get("gene", ""),
                        g.get("product", ""),
                        g.get("strand", 1),
                        g["end"] - g["start"],
                        distance
                    ])

    def write_indicator_genes_csv(self, contig_id, indicator_genes, output_csv):
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["contig_id", "start", "end", "gene", "product", "locus_tag", "strand", "indicator_gene"])
            for gene in indicator_genes:
                writer.writerow([
                    contig_id,
                    gene["start"],
                    gene["end"],
                    gene.get("gene", ""),
                    gene.get("product", ""),
                    gene.get("locus_tag", ""),
                    gene.get("strand", 1),
                    gene.get("indicator_gene", "")
                ])

    def save_results_csv(self, results, output_path):
        with open(output_path, "w") as f:
            f.write("SequenceID,Length,Region_Start,Region_End,Probability,Sequence\n")
            for result in results:
                for region in result["regions"]:
                    if region.get("max_prob", 0) >= THRESHOLD:
                        sequence = region.get("sequence", "")
                        f.write(f"{result['id']},{result['length']},{region['start']},{region['end']},{region['max_prob']:.4f},{sequence}\n")

    def save_html_report(self, results, global_prob_img_paths, dnaa_img_paths, output_path):
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<meta charset='UTF-8'>")
        html.append("<title>MAG Replication Origin Prediction Report</title>")
        html.append("<style>")
        html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html.append("h1, h2, h3 { color: #333; }")
        html.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append(".params { background-color: #e9ecef; padding: 10px; margin: 10px 0; }")
        html.append(".region-block { margin: 20px 0; padding: 15px; border: 1px solid #ddd; }")
        html.append("img { max-width: 100%; height: auto; margin: 10px 0; }")
        html.append(".sequence-pre { overflow: auto; max-height: 400px; white-space: pre-wrap; word-wrap: break-word; background-color: #f8f9fa; padding: 10px; border: 1px solid #ddd; font-family: monospace; font-size: 12px; }")
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        html.append("<h1>MAG Replication Origin Prediction Report</h1>")
        html.append("<div class='params'>")
        html.append("<h3>Prediction Parameters</h3>")
        html.append(f"<p><strong>Sequence Length:</strong> {SEQ_LENGTH} bp</p>")
        html.append(f"<p><strong>Step Size:</strong> {STEP_SIZE} bp</p>")
        html.append(f"<p><strong>Probability Threshold:</strong> {THRESHOLD}</p>")
        html.append("</div>")
        total_contigs = len(results)
        contigs_with_predictions = len([r for r in results if r.get("regions")])
        html.append("<div class='params'>")
        html.append("<h3>Summary Statistics</h3>")
        html.append(f"<p><strong>Total Contigs Analyzed:</strong> {total_contigs}</p>")
        html.append(f"<p><strong>Contigs with OriC Predictions:</strong> {contigs_with_predictions}</p>")
        html.append("</div>")
        for result in results:
            if not result.get("regions"):
                continue
            seq_id = result["id"]
            html.append(f"<h2>Sequence: {seq_id} (Length: {result['length']:,} bp)</h2>")
            if seq_id in global_prob_img_paths:
                html.append("<h3>Global Prediction Probability</h3>")
                with open(global_prob_img_paths[seq_id], "rb") as imgf:
                    img_b64 = base64.b64encode(imgf.read()).decode("utf-8")
                html.append(f"<img src='data:image/png;base64,{img_b64}' alt='Global Probability'>")
            if seq_id in dnaa_img_paths:
                html.append("<h3>DnaA Box Count</h3>")
                with open(dnaa_img_paths[seq_id], "rb") as imgf:
                    img_b64 = base64.b64encode(imgf.read()).decode("utf-8")
                html.append(f"<img src='data:image/png;base64,{img_b64}' alt='DnaA Box Count'>")
            html.append("<h3>Predicted Origin Regions</h3>")
            html.append("<table>")
            html.append("<tr><th>Rank</th><th>Start</th><th>End</th><th>Center</th><th>Probability</th></tr>")
            for i, region in enumerate(result["regions"], 1):
                center = (region["start"] + region["end"]) // 2
                html.append("<tr>")
                html.append(f"<td>{i}</td>")
                html.append(f"<td>{region['start']:,}</td>")
                html.append(f"<td>{region['end']:,}</td>")
                html.append(f"<td>{center:,}</td>")
                html.append(f"<td>{region['max_prob']:.4f}</td>")
                html.append("</tr>")
            html.append("</table>")
            html.append("<h3>Gene Structure Analysis for Each Predicted Region</h3>")
            for i, region in enumerate(result["regions"], 1):
                html.append(f"<div class='region-block'>")
                html.append(f"<h4>Region {i}: {region['start']:,} - {region['end']:,} (Probability: {region['max_prob']:.4f})</h4>")
                if "structure_img" in region and os.path.exists(region["structure_img"]):
                    html.append("<h5>Gene Structure (10kb upstream and downstream from OriC center)</h5>")
                    with open(region["structure_img"], "rb") as imgf:
                        img_b64 = base64.b64encode(imgf.read()).decode("utf-8")
                    html.append(f"<img src='data:image/png;base64,{img_b64}' alt='Gene Structure'>")
                if "sequence" in region and len(region["sequence"]) > 0:
                    html.append("<h5>Region Sequence:</h5>")
                    # Display full sequence with scrollable box
                    html.append(f"<pre class='sequence-pre'>{region['sequence']}</pre>")
                html.append("</div>")
            html.append("<hr>")
        html.append("<p><em>Generated by DeepBacOri MAG Predictor</em></p>")
        html.append("</body>")
        html.append("</html>")
        with open(output_path, "w") as f:
            f.write("\n".join(html))


class MAGPredictor:
    def __init__(self, output_dir, threshold=THRESHOLD, verbose=False, use_gpu=False):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.threshold = threshold
        self.verbose = verbose
        self.dl_predictor = DeepLearningPredictor(use_gpu=use_gpu)
        self.dnaa_analyzer = DnaABoxAnalyzer()
        self.gene_analyzer = GeneStructureAnalyzer(verbose=verbose, output_dir=output_dir)
        self.exporter = ResultExporter()
        self.images_created = 0

    def _extract_contig_id(self, header):
        if not header:
            return ""
        h = str(header).lstrip(">").strip()
        token = h.split()[0]
        token = re.sub(r"(?:\.\d+)+$", "", token)
        token = re.sub(r"/\d+$", "", token)
        return token

    def predict(self, fasta_path, gene_file=None):
        results = []
        global_prob_img_paths = {}
        dnaa_img_paths = {}
        base = os.path.splitext(os.path.splitext(os.path.basename(fasta_path))[0])[0]

        self.gene_analyzer.setup_log_file(base)
        start_time = time.time()
        self.gene_analyzer._log(f"Starting analysis for MAG: {base}", to_console=self.verbose)
        self.gene_analyzer._log(f"FASTA file: {fasta_path}")
        self.gene_analyzer._log(f"Gene file: {gene_file}")
        self.gene_analyzer._log(f"Output directory: {self.output_dir}")
        self.gene_analyzer._log("="*60)

        if gene_file and os.path.exists(gene_file):
            self.gene_analyzer._log(f"Preloading GBK cache...")
            self.gene_analyzer.load_gbk_cache(gene_file)

        opener = gzip.open if fasta_path.endswith(".gz") else open
        with opener(fasta_path, "rt") as handle:
            fasta_records = list(SeqIO.parse(handle, "fasta"))
            self.gene_analyzer._log(f"FASTA contains {len(fasta_records)} contigs")
            for record_idx, record in enumerate(fasta_records):
                seq = str(record.seq)
                seq_id = self._extract_contig_id(record.id)
                original_header = record.id

                self.gene_analyzer._log(f"\n{'='*60}")
                self.gene_analyzer._log(f"Processing contig {record_idx + 1}/{len(fasta_records)}")
                self.gene_analyzer._log(f"Genome contig ID (normalized): {seq_id}")
                self.gene_analyzer._log(f"Contig length: {len(seq):,} bp")

                if len(seq) < SEQ_LENGTH:
                    self.gene_analyzer._log(f"Contig too short for prediction, skipping")
                    continue

                self.gene_analyzer._log(f"Running deep learning prediction...")
                dl_positions, dl_probs = self.dl_predictor.predict_sequence(seq)
                dl_peak_positions, dl_peak_probs = self.dl_predictor.find_peaks(dl_positions, dl_probs, self.threshold)

                if not dl_peak_positions:
                    self.gene_analyzer._log(f"No prediction peaks found above threshold, skipping contig")
                    continue

                self.gene_analyzer._log(f"Found {len(dl_peak_positions)} prediction peaks above threshold")

                genes = []
                if gene_file and os.path.exists(gene_file):
                    self.gene_analyzer._log(f"Fetching genes for contig '{seq_id}'...")
                    genes = self.gene_analyzer.get_genes_for_contig(gene_file, seq_id)
                    if not genes:
                        self.gene_analyzer._log(f"Trying with original header '{original_header}'...")
                        genes = self.gene_analyzer.get_genes_for_contig(gene_file, original_header)

                global_prob_path = os.path.join(self.output_dir, f"{base}_{seq_id}_global_prob.png")
                prob_img_path = self.dl_predictor.plot_global_probability(dl_positions, dl_probs, seq_id, global_prob_path)
                global_prob_img_paths[seq_id] = prob_img_path
                self.images_created += 1

                dnaa_boxes, dnaa_img = self.dnaa_analyzer.run_dnaa_box_analysis_optimized(seq, seq_id, self.output_dir)
                if dnaa_img:
                    dnaa_img_paths[seq_id] = dnaa_img
                    self.images_created += 1

                indicator_genes = []
                if genes:
                    indicator_genes = self.gene_analyzer.find_indicator_genes(genes)
                    if indicator_genes:
                        indicator_csv = os.path.join(self.output_dir, f"{base}_{seq_id}_indicator_genes.csv")
                        self.exporter.write_indicator_genes_csv(seq_id, indicator_genes, indicator_csv)
                        self.gene_analyzer._log(f"Found {len(indicator_genes)} indicator genes")

                final_regions = []
                for pos_idx, (pos, prob) in enumerate(zip(dl_peak_positions, dl_peak_probs)):
                    center_pos = pos + (SEQ_LENGTH // 2)
                    start = max(0, center_pos - SEQ_LENGTH // 2)
                    end = min(len(seq), center_pos + SEQ_LENGTH // 2)
                    region_data = {"start": int(start), "end": int(end), "max_prob": float(prob), "sequence": seq[start:end]}

                    structure_img = os.path.join(self.output_dir, f"{base}_{seq_id}_structure_{region_data['start']}_{region_data['end']}.png")
                    self.gene_analyzer.plot_gene_structure_optimized(seq, genes, region_data["start"], region_data["end"], structure_img)
                    region_data["structure_img"] = structure_img
                    self.images_created += 1

                    if genes:
                        flanking_csv = os.path.join(self.output_dir, f"{base}_{seq_id}_flanking10kb_genes.csv")
                        flanking_genes = self.gene_analyzer.get_flanking_genes(genes, region_data["start"], region_data["end"], flanking_distance=10000)
                        if flanking_genes:
                            self.exporter.write_flanking10kb_genes_csv(seq_id, [region_data], flanking_genes, flanking_csv)
                            self.gene_analyzer._log(f"Region {pos_idx+1}: Saved {len(flanking_genes)} flanking genes to CSV")
                        else:
                            self.exporter.write_flanking10kb_genes_csv(seq_id, [], [], flanking_csv)
                            self.gene_analyzer._log(f"Region {pos_idx+1}: No flanking genes found")

                    final_regions.append(region_data)

                results.append({
                    "id": seq_id,
                    "original_header": original_header,
                    "length": len(seq),
                    "regions": final_regions,
                    "has_genes": len(genes) > 0,
                    "num_genes": len(genes)
                })
                self.gene_analyzer._log(f"Added {len(final_regions)} predicted regions for contig '{seq_id}'")

        elapsed_time = time.time() - start_time
        self.gene_analyzer._log(f"\n{'='*60}")
        self.gene_analyzer._log(f"Analysis complete for MAG: {base}")
        self.gene_analyzer._log(f"Results: {len(results)} contigs with predictions")
        self.gene_analyzer._log(f"Images created: {self.images_created}")
        self.gene_analyzer._log(f"Time elapsed: {elapsed_time:.2f} seconds")
        self.gene_analyzer._log(f"Log file saved to: {self.gene_analyzer.log_file}")

        return results, global_prob_img_paths, dnaa_img_paths, self.images_created

    def save_results(self, results, output_path):
        self.exporter.save_results_csv(results, output_path)

    def save_html_report(self, results, global_prob_img_paths, dnaa_img_paths, output_path):
        self.exporter.save_html_report(results, global_prob_img_paths, dnaa_img_paths, output_path)


def find_gene_file(fasta_path, gene_dir):
    if not gene_dir or not os.path.exists(gene_dir):
        return None

    base_name = os.path.basename(fasta_path)
    base_no_ext = base_name
    while True:
        name, ext = os.path.splitext(base_no_ext)
        if ext in ['.fna', '.fa', '.fasta', '.ffn', '.faa', '.gz', '.gzip']:
            base_no_ext = name
        else:
            break

    possible_gene_files = []
    for ext in ['.gbff.gz', '.gbk.gz', '.gbff', '.gbk', '.gb', '.gff', '.gff.gz']:
        candidate = os.path.join(gene_dir, base_no_ext + ext)
        if os.path.exists(candidate):
            possible_gene_files.append(candidate)

    if not possible_gene_files:
        import re
        accession_pattern = re.search(r'([A-Z]{3}_[0-9]+\.[0-9]+)', base_no_ext)
        if accession_pattern:
            accession = accession_pattern.group(1)
            for file in os.listdir(gene_dir):
                if accession in file and any(file.endswith(ext) for ext in ['.gbff.gz', '.gbk.gz', '.gbff', '.gbk', '.gb', '.gff', '.gff.gz']):
                    candidate = os.path.join(gene_dir, file)
                    possible_gene_files.append(candidate)

    if possible_gene_files:
        for file in possible_gene_files:
            if not file.endswith('.gz'):
                return file
        return possible_gene_files[0]
    return None


def process_mag_genome(fasta_path, gene_file, output_dir, threshold=THRESHOLD, verbose=False):
    base = os.path.splitext(os.path.splitext(os.path.basename(fasta_path))[0])[0]
    predictor = MAGPredictor(output_dir=output_dir, threshold=threshold, verbose=verbose, use_gpu=False)
    try:
        results, global_prob_img_paths, dnaa_img_paths, images_created = predictor.predict(fasta_path, gene_file=gene_file)
        if results:
            output_csv = os.path.join(output_dir, f"{base}_predictions.csv")
            output_html = os.path.join(output_dir, f"{base}_report.html")
            predictor.save_results(results, output_csv)
            predictor.save_html_report(results, global_prob_img_paths, dnaa_img_paths, output_html)
        contigs_with_predictions = len(results)
        total_contigs = sum(1 for _ in SeqIO.parse(gzip.open(fasta_path, "rt") if fasta_path.endswith(".gz") else open(fasta_path), "fasta"))
        if contigs_with_predictions > 0:
            return f"{base}: {contigs_with_predictions}/{total_contigs} contigs have predictions ({images_created} images created)"
        else:
            return f"{base}: No predictions found (0/{total_contigs} contigs)"
    except Exception as e:
        import traceback
        error_msg = f"{base} failed: {str(e)}\n{traceback.format_exc()}"
        log_file = os.path.join(output_dir, f"{base}_error.log")
        with open(log_file, "w") as f:
            f.write(error_msg)
        return f"{base} failed: {str(e)}"


def _signal_handler(signum, frame):
    global _shutdown_flag
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    _shutdown_flag = True
    # Force exit after a short grace period
    time.sleep(2)
    sys.exit(1)


def batch_process_mags(fasta_files, gene_file_map, output_dir, threshold, verbose, batch_size_mags=10, max_workers=2):
    total_mags = len(fasta_files)
    processed = 0
    global _shutdown_flag

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    for i in range(0, total_mags, batch_size_mags):
        if _shutdown_flag:
            print("Shutdown requested, terminating batch processing.")
            break

        batch_files = fasta_files[i:i+batch_size_mags]
        batch_num = i // batch_size_mags + 1
        total_batches = (total_mags + batch_size_mags - 1) // batch_size_mags
        print(f"\n{'='*60}")
        print(f"Processing batch {batch_num}/{total_batches}")
        print(f"Batch size: {len(batch_files)} MAGs")
        print(f"Progress: {processed}/{total_mags} MAGs processed")
        print(f"{'='*60}")

        batch_start_time = time.time()
        executor = None
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
            futures = []
            for fasta_path in batch_files:
                gene_file = gene_file_map.get(fasta_path)
                futures.append(executor.submit(process_mag_genome, fasta_path, gene_file, output_dir, threshold, verbose))

            # Iterate with timeout to avoid hanging
            for fut in as_completed(futures, timeout=3600):  # 1 hour timeout per MAG
                if _shutdown_flag:
                    break
                try:
                    result = fut.result(timeout=300)  # 5 min per future
                    print(result)
                    processed += 1
                except TimeoutError:
                    print("A task timed out, continuing...")
                except Exception as e:
                    print(f"Task raised exception: {e}")
        except KeyboardInterrupt:
            print("\nInterrupted by user. Terminating workers...")
            if executor:
                executor.shutdown(wait=False, cancel_futures=True)
                # Force kill remaining processes (optional but effective)
                for pid in executor._processes.keys():
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception:
                        pass
            raise
        finally:
            if executor:
                executor.shutdown(wait=True)
        batch_time = time.time() - batch_start_time
        print(f"Batch {batch_num} completed in {batch_time:.2f} seconds")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(1)
    return processed


def main():
    parser = argparse.ArgumentParser(description="Predict DNA replication origin sites in MAGs (Optimized with batch processing)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Input FASTA file path (single MAG)")
    group.add_argument("-d", "--input_dir", help="Input directory containing multiple FASTA files for batch prediction")
    parser.add_argument("-g", "--gene_file", help="GBK/GBFF gene annotation file (optional)")
    parser.add_argument("-gd", "--gene_dir", help="Directory containing GBK/GBFF gene files for batch mode")
    parser.add_argument("-o", "--output", help="Output directory for all results (default: MAG_predictions_optimized)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help=f"Probability threshold for displaying results (default: {THRESHOLD})")
    parser.add_argument("--batch_size_mags", type=int, default=10, help=f"Number of MAGs to process in each batch (default: 10)")
    parser.add_argument("--max_workers", type=int, default=2, help=f"Maximum number of worker processes (default: 2)")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for prediction (use with caution for large batches)")
    parser.add_argument("--verbose", action="store_true", help="Verbose output for debugging contig/GBK matching")
    args = parser.parse_args()

    output_dir = args.output or "MAG_predictions_optimized"
    os.makedirs(output_dir, exist_ok=True)

    if args.input_dir:
        import glob
        fasta_files = []
        for ext in ("*.fa", "*.fasta", "*.fna", "*.fna.gz", "*.ffn", "*.faa"):
            fasta_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        if not fasta_files:
            print("No FASTA files found in input directory")
            return
        print(f"Found {len(fasta_files)} MAGs in input directory")
        print(f"Output directory: {output_dir}")
        print(f"Batch size: {args.batch_size_mags} MAGs per batch")
        print(f"Max workers: {args.max_workers}")
        if args.verbose:
            print(f"Verbose mode: ON")

        gene_file_map = {}
        if args.gene_dir:
            print(f"Searching for gene files in: {args.gene_dir}")
            for fasta_path in fasta_files:
                gene_file = find_gene_file(fasta_path, args.gene_dir)
                gene_file_map[fasta_path] = gene_file
                if gene_file:
                    print(f"  {os.path.basename(fasta_path)} -> {os.path.basename(gene_file)}")
                else:
                    print(f"  {os.path.basename(fasta_path)} -> No gene file found")
        else:
            for fasta_path in fasta_files:
                gene_file_map[fasta_path] = None

        total_start_time = time.time()
        processed = batch_process_mags(
            fasta_files, gene_file_map, output_dir,
            args.threshold, args.verbose, args.batch_size_mags, args.max_workers
        )
        total_time = time.time() - total_start_time
        print(f"\n{'='*60}")
        print(f"All batches completed!")
        print(f"Total MAGs processed: {processed}/{len(fasta_files)}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Average time per MAG: {total_time/len(fasta_files):.2f} seconds")
    else:
        base = os.path.splitext(os.path.splitext(os.path.basename(args.input))[0])[0]
        gene_file = args.gene_file
        if not gene_file and hasattr(args, 'gene_dir') and args.gene_dir:
            gene_file = find_gene_file(args.input, args.gene_dir)
        print(f"Starting prediction for MAG: {base}")
        print(f"Input FASTA: {args.input}")
        if gene_file:
            print(f"Gene file: {gene_file}")
        else:
            print(f"No gene file provided, skipping gene analysis")
        print(f"Output directory: {output_dir}")
        if args.verbose:
            print(f"Verbose mode: ON")

        start_time = time.time()
        predictor = MAGPredictor(output_dir=output_dir, threshold=args.threshold, verbose=args.verbose, use_gpu=args.use_gpu)
        results, global_prob_img_paths, dnaa_img_paths, images_created = predictor.predict(args.input, gene_file=gene_file)
        contigs_with_predictions = len(results)
        total_contigs = sum(1 for _ in SeqIO.parse(gzip.open(args.input, "rt") if args.input.endswith(".gz") else open(args.input), "fasta"))
        elapsed_time = time.time() - start_time
        print(f"\nPrediction completed for {base}")
        print(f"Found predictions in {contigs_with_predictions}/{total_contigs} contigs")
        print(f"Images created: {images_created}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        if results:
            output_csv = os.path.join(output_dir, f"{base}_predictions.csv")
            output_html = os.path.join(output_dir, f"{base}_report.html")
            predictor.save_results(results, output_csv)
            predictor.save_html_report(results, global_prob_img_paths, dnaa_img_paths, output_html)
            print(f"Results saved to: {output_csv}")
            print(f"HTML report saved to: {output_html}")
        else:
            print(f"No predictions found, skipping CSV and HTML report creation")


if __name__ == "__main__":
    main()