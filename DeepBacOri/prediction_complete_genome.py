"""
prediction.py
Predict replication origin sites in complete genomes using trained model
"""
import os
import argparse
import numpy as npcomplete
import torch
from Bio import SeqIO
import matplotlib.pyplot as plt
import base64
from scipy.signal import find_peaks
import matplotlib.ticker as ticker
import csv
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed

DEFAULT_MODEL_PATH = "model/model.pth"
DEFAULT_SEQ_LENGTH = 2000
DEFAULT_STEP_SIZE = 500
DEFAULT_BATCH_SIZE = 64
DEFAULT_THRESHOLD = 0.8
DEFAULT_GC_WINDOW_SIZE = 10000
DEFAULT_GC_STEP_SIZE = 1000
DEFAULT_OVERLAP_THRESHOLD = 500
DNAA_BOX_FORWARD = "TTATCCACA"
DNAA_BOX_REVERSE = "TGTGGATAA"

def calculate_circular_distance(pos1, pos2, seq_length):
    d = abs(pos1 - pos2)
    return min(d, seq_length - d)

def calculate_z_curve_components(sequence, window_size=10000, step_size=1000):
    if len(sequence) < window_size:
        return [], {'x_component': [], 'y_component': [], 'z_component': []}
    positions = []
    x_component = []
    y_component = []
    z_component = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        window = sequence[i:i + window_size]
        a_count = window.count('A')
        t_count = window.count('T')
        g_count = window.count('G')
        c_count = window.count('C')
        x = (a_count + g_count) - (c_count + t_count)
        y = (a_count + c_count) - (g_count + t_count)
        z = (a_count + t_count) - (g_count + c_count)
        x_component.append(x / window_size)
        y_component.append(y / window_size)
        z_component.append(z / window_size)
        positions.append(i + window_size // 2)
    return positions, {'x_component': x_component, 'y_component': y_component, 'z_component': z_component}

def find_potential_oric_positions(positions, cumulative_components):
    potential_positions = []
    extrema_info = []
    for component_name, component in cumulative_components.items():
        peaks, _ = find_peaks(component)
        troughs, _ = find_peaks(-component)
        extrema = np.sort(np.concatenate([peaks, troughs]))
        for idx in extrema:
            potential_positions.append(positions[idx])
            extrema_info.append({'position': positions[idx], 'component': component_name,
                                 'value': component[idx], 'type': 'peak' if idx in peaks else 'trough'})
    return potential_positions, extrema_info

def predict_oric_regions(sequence, potential_positions, min_region_size=500, max_region_size=2000):
    regions = []
    potential_positions = sorted(potential_positions)
    current_group = []
    for pos in potential_positions:
        if not current_group or pos - current_group[-1] <= max_region_size:
            current_group.append(pos)
        else:
            if len(current_group) >= 2:
                regions.append({'start': max(0, current_group[0] - max_region_size // 2),
                                'end': min(len(sequence), current_group[-1] + max_region_size // 2),
                                'positions': current_group.copy()})
            current_group = [pos]
    if len(current_group) >= 2:
        regions.append({'start': max(0, current_group[0] - max_region_size // 2),
                        'end': min(len(sequence), current_group[-1] + max_region_size // 2),
                        'positions': current_group})
    return regions

def score_oric_regions(regions, extrema_info, sequence_length):
    scored_regions = []
    for region in regions:
        region_extrema = [info for info in extrema_info if region['start'] <= info['position'] <= region['end']]
        score = min(0.95, 0.3 + 0.1 * len(region_extrema))
        wraps_around = region['end'] > sequence_length
        scored_region = {'start': region['start'], 'end': region['end'] % sequence_length if wraps_around else region['end'],
                         'length': region['end'] - region['start'], 'score': score, 'wraps_around': wraps_around,
                         'extrema_count': len(region_extrema)}
        if wraps_around:
            scored_region['second_end'] = region['end']
        scored_regions.append(scored_region)
    scored_regions.sort(key=lambda x: x['score'], reverse=True)
    return scored_regions

def find_dnaa_boxes(sequence, patterns=None, max_mismatches=1, include_reverse=False):
    if patterns is None:
        patterns = {"DnaA_box": DNAA_BOX_FORWARD}
    boxes = []
    pattern_length = len(DNAA_BOX_FORWARD)
    for pattern_name, pattern_seq in patterns.items():
        for i in range(len(sequence) - pattern_length + 1):
            segment = sequence[i:i+pattern_length]
            mismatches = sum(1 for a, b in zip(segment.upper(), pattern_seq.upper()) if a != b)
            if mismatches <= max_mismatches:
                score = 1.0 - (mismatches / pattern_length)
                boxes.append({'start': i, 'end': i + pattern_length, 'sequence': segment,
                              'pattern': pattern_name, 'mismatches': mismatches, 'score': score})
        if include_reverse:
            rev_pattern = DNAA_BOX_REVERSE
            for i in range(len(sequence) - pattern_length + 1):
                segment = sequence[i:i+pattern_length]
                mismatches = sum(1 for a, b in zip(segment.upper(), rev_pattern.upper()) if a != b)
                if mismatches <= max_mismatches:
                    score = 1.0 - (mismatches / pattern_length)
                    boxes.append({'start': i, 'end': i + pattern_length, 'sequence': segment,
                                  'pattern': pattern_name + "_rev", 'mismatches': mismatches, 'score': score})
    boxes.sort(key=lambda x: x['start'])
    return boxes

def calculate_dnaa_box_density(sequence, boxes, window_size=1000, step_size=100):
    positions = []
    densities = []
    for i in range(0, len(sequence) - window_size + 1, step_size):
        window_end = i + window_size
        window_boxes = [box for box in boxes if (i <= box['start'] < window_end) or (i <= box['end'] < window_end)]
        if window_boxes:
            total_score = sum(box['score'] for box in window_boxes)
            density = (total_score * 1000) / window_size
        else:
            density = 0.0
        positions.append(i + window_size // 2)
        densities.append(density)
    return positions, densities

def find_dnaa_clusters(boxes, max_distance=500):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x['start'])
    clusters = []
    current_cluster = [boxes[0]]
    for box in boxes[1:]:
        if box['start'] - current_cluster[-1]['end'] <= max_distance:
            current_cluster.append(box)
        else:
            if len(current_cluster) >= 2:
                clusters.append(current_cluster)
            current_cluster = [box]
    if len(current_cluster) >= 2:
        clusters.append(current_cluster)
    cluster_dicts = []
    for i, cluster in enumerate(clusters):
        start = min(box['start'] for box in cluster)
        end = max(box['end'] for box in cluster)
        box_count = len(cluster)
        avg_score = sum(box['score'] for box in cluster) / box_count
        cluster_length = end - start
        density_bonus = min(1.0, 3000 / max(1000, cluster_length))
        score = min(0.99, (0.3 * box_count / 10) + (0.4 * avg_score) + (0.3 * density_bonus))
        cluster_dicts.append({'start': start, 'end': end, 'length': end - start, 'boxes': cluster,
                              'box_count': box_count, 'avg_box_score': avg_score, 'score': score, 'rank': i + 1})
    cluster_dicts.sort(key=lambda x: x['score'], reverse=True)
    return cluster_dicts

def predict_oric_regions_from_clusters(sequence, clusters, min_region_size=500, max_region_size=2000):
    regions = []
    genome_length = len(sequence)
    for i, cluster in enumerate(clusters[:5]):
        start = cluster['start']
        end = cluster['end']
        if (end - start) < min_region_size:
            center = (start + end) // 2
            half_size = min_region_size // 2
            start = max(0, center - half_size)
            end = min(genome_length, center + half_size)
        if (end - start) > max_region_size:
            center = (start + end) // 2
            half_size = max_region_size // 2
            start = max(0, center - half_size)
            end = min(genome_length, center + half_size)
        regions.append({'start': start, 'end': end, 'length': end - start, 'box_count': cluster['box_count'],
                        'score': cluster['score'], 'wraps_around': False, 'rank': i + 1})
    regions.sort(key=lambda x: x['score'], reverse=True)
    return regions

class DNAPredictor:
    class DNAModel(torch.nn.Module):
        def __init__(self, seq_length=2000, n_features=4):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(n_features, 64, kernel_size=9)
            self.pool = torch.nn.MaxPool1d(2)
            self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=9)
            self.global_pool = torch.nn.AdaptiveMaxPool1d(1)
            self.fc1 = torch.nn.Linear(128, 64)
            self.dropout = torch.nn.Dropout(0.35)
            self.fc2 = torch.nn.Linear(64, 1)
        def forward(self, x):
            x = x.permute(0, 2, 1)
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = self.global_pool(x).squeeze(-1)
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)
    
    def __init__(self, output_dir="DeepBacOri_predictions", model_path=DEFAULT_MODEL_PATH,
                 seq_length=DEFAULT_SEQ_LENGTH, step_size=DEFAULT_STEP_SIZE,
                 batch_size=DEFAULT_BATCH_SIZE, threshold=DEFAULT_THRESHOLD,
                 gc_window=DEFAULT_GC_WINDOW_SIZE, gc_step=DEFAULT_GC_STEP_SIZE,
                 overlap_threshold=DEFAULT_OVERLAP_THRESHOLD):
        self.output_dir = output_dir
        self.model_path = model_path
        self.seq_length = seq_length
        self.step_size = step_size
        self.batch_size = batch_size
        self.threshold = threshold
        self.gc_window = gc_window
        self.gc_step = gc_step
        self.overlap_threshold = overlap_threshold
        os.makedirs(self.output_dir, exist_ok=True)
        self.device, self.model = self.load_model()
        self._extra_results = {}
    
    def load_model(self):
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = self.DNAModel().to(device)
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            return device, model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def preprocess(self, seq):
        seq = seq.upper()
        seq = ''.join(['ATCG'[np.random.randint(4)] if nt == 'N' else nt for nt in seq])
        encoded = np.zeros((self.seq_length, 4), dtype=np.float32)
        for i, nt in enumerate(seq[:self.seq_length]):
            if nt == 'A': encoded[i, 0] = 1
            elif nt == 'T': encoded[i, 1] = 1
            elif nt == 'C': encoded[i, 2] = 1
            elif nt == 'G': encoded[i, 3] = 1
        return encoded
    
    def calculate_gc_skew(self, seq):
        if len(seq) < self.gc_window:
            return [], []
        gc_skew = []
        positions = []
        for i in range(0, len(seq) - self.gc_window + 1, self.gc_step):
            window = seq[i:i + self.gc_window]
            g_count = window.count('G')
            c_count = window.count('C')
            if g_count + c_count > 0:
                skew = (g_count - c_count) / (g_count + c_count)
            else:
                skew = 0
            gc_skew.append(skew)
            positions.append(i + self.gc_window // 2)
        return positions, gc_skew
    
    def find_peaks(self, positions, probs, threshold=None):
        if threshold is None:
            threshold = self.threshold
        peaks = []
        peak_probs = []
        for i in range(len(positions)):
            if probs[i] > threshold:
                peaks.append(positions[i])
                peak_probs.append(probs[i])
        return peaks, peak_probs
    
    def merge_overlapping_predictions(self, regions, full_sequence=None):
        if not regions:
            return regions
        sorted_regions = sorted(regions, key=lambda x: x['start'])
        merged = []
        current = sorted_regions[0].copy()
        for next_region in sorted_regions[1:]:
            if next_region['start'] <= current['end'] + self.overlap_threshold:
                current['end'] = max(current['end'], next_region['end'])
                current['max_prob'] = max(current['max_prob'], next_region['max_prob'])
            else:
                if full_sequence is not None:
                    current['sequence'] = full_sequence[current['start']:current['end']]
                merged.append(current)
                current = next_region.copy()
        if full_sequence is not None:
            current['sequence'] = full_sequence[current['start']:current['end']]
        merged.append(current)
        return merged
    
    def plot_gc_skew_circular(self, fasta_basename, seq_id, gc_positions, gc_skew, seq_length, dl_peaks=None):
        if not gc_skew or len(gc_skew) == 0:
            img_path = os.path.join(self.output_dir, f"{fasta_basename}_{seq_id}_gcskew_circular.png")
            self.plot_placeholder(img_path, "No GC skew data available")
            return img_path
        gc_skew = np.array(gc_skew)
        max_abs = np.max(np.abs(gc_skew))
        if max_abs == 0: max_abs = 1
        norm_gc_skew = gc_skew / max_abs
        n = len(gc_skew)
        theta = 2 * np.pi * np.arange(n) / n
        width = 2 * np.pi / n
        fig, ax = plt.subplots(figsize=(7,7), subplot_kw={'polar': True})
        for i in range(n):
            color = 'green' if norm_gc_skew[i] >= 0 else 'purple'
            r0 = 1.0
            r1 = 1.0 + abs(norm_gc_skew[i]) * 0.5
            ax.bar(theta[i], r1 - r0, width=width, bottom=r0, color=color, edgecolor=None, linewidth=0)
        tick_step = 1_000_000
        tick_positions = np.arange(0, seq_length+1, tick_step)
        if tick_positions[-1] != seq_length:
            tick_positions = np.append(tick_positions, seq_length)
        for pos in tick_positions:
            theta_tick = 2 * np.pi * (pos / seq_length)
            label = f"{pos/1e6:.1f} Mbp"
            ax.plot([theta_tick, theta_tick], [0.9, 1.6], color='gray', lw=1, alpha=0.5)
            ax.text(theta_tick, 1.65, label, ha='center', va='center', fontsize=11)
        if dl_peaks:
            for i, peak in enumerate(dl_peaks):
                theta_peak = 2 * np.pi * (peak / seq_length)
                if i == 0:
                    ax.plot([theta_peak], [1.6], marker='o', color='red', markersize=12, label='DL Prediction')
                else:
                    ax.plot([theta_peak], [1.6], marker='o', color='red', markersize=12)
            ax.legend(loc='upper right')
        ax.set_axis_off()
        ax.set_ylim(0, 1.7)
        ax.set_title(f'GC Skew Circular Plot: {seq_id}', va='bottom', fontsize=14)
        img_path = os.path.join(self.output_dir, f"{fasta_basename}_{seq_id}_gcskew_circular.png")
        plt.tight_layout()
        plt.savefig(img_path, dpi=300)
        plt.close()
        return img_path
    
    def plot_sci_zcurve(self, positions, x_cum, y_cum, z_cum, seq_length, seq_id, save_path):
        if not positions:
            self.plot_placeholder(save_path, "No Z-curve data")
            return
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ln1 = ax1.plot(np.array(positions)/1000, x_cum, color='red', linewidth=2, label=r'$x_n$')
        ax1.set_ylabel(r'$x_n$', color='red', fontsize=14)
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.set_xlabel('Sequence length (kb)', fontsize=14)
        ax2 = ax1.twinx()
        ln2 = ax2.plot(np.array(positions)/1000, y_cum, color='blue', linewidth=2, label=r'$y_n$')
        ax2.set_ylabel(r'$y_n$', color='blue', fontsize=14)
        ax2.tick_params(axis='y', labelcolor='blue')
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 60))
        ln3 = ax3.plot(np.array(positions)/1000, z_cum, color='green', linewidth=2, label=r'$G C_n$')
        ax3.set_ylabel(r'$G C_n$', color='green', fontsize=14)
        ax3.tick_params(axis='y', labelcolor='green')
        lns = ln1 + ln2 + ln3
        labels = [l.get_label() for l in lns]
        ax1.legend(lns, labels, loc='upper left', fontsize=12)
        ax1.set_title(f'Z-curve: {seq_id}', fontsize=16)
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    def run_z_curve_analysis(self, seq, seq_id, fasta_basename):
        positions, components = calculate_z_curve_components(seq, self.gc_window, self.gc_step)
        if not positions:
            plot_path = os.path.join(self.output_dir, f"{fasta_basename}_{seq_id}_zcurve.png")
            self.plot_placeholder(plot_path, "No Z-curve data")
            return [], plot_path
        cumulative_components = {'x_component': np.cumsum(components['x_component']),
                                 'y_component': np.cumsum(components['y_component']),
                                 'z_component': np.cumsum(components['z_component'])}
        potential_positions, extrema_info = find_potential_oric_positions(positions, cumulative_components)
        regions = predict_oric_regions(seq, potential_positions, min_region_size=500, max_region_size=2000)
        scored_regions = score_oric_regions(regions, extrema_info, len(seq))
        plot_path = os.path.join(self.output_dir, f"{fasta_basename}_{seq_id}_zcurve.png")
        self.plot_sci_zcurve(positions, cumulative_components['x_component'],
                             cumulative_components['y_component'], cumulative_components['z_component'],
                             len(seq), seq_id, plot_path)
        return scored_regions, plot_path
    
    def run_dnaa_box_analysis(self, seq, seq_id, fasta_basename):
        boxes = find_dnaa_boxes(seq, patterns=None, max_mismatches=1, include_reverse=False)
        positions, densities = calculate_dnaa_box_density(seq, boxes, window_size=1000, step_size=100)
        clusters = find_dnaa_clusters(boxes)
        regions = predict_oric_regions_from_clusters(seq, clusters, min_region_size=500, max_region_size=2000)
        plot_path = os.path.join(self.output_dir, f"{fasta_basename}_{seq_id}_dnaa_box.png")
        plt.figure(figsize=(15, 5))
        plt.plot(positions, densities, label='DnaA Box Density')
        for region in regions:
            plt.axvspan(region['start'], region['end'], alpha=0.2, color='red')
        plt.title('DnaA Box Density and Predicted oriC Regions')
        plt.xlabel('Position')
        plt.ylabel('Density (boxes/kb)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        return regions, plot_path
    
    def plot_global_probability(self, fasta_basename, seq_id, dl_positions, dl_probs):
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(dl_positions, dl_probs, color='#1f77b4', linewidth=2, label='Prediction Probability')
        ax.set_xlabel('Genome Position', fontsize=14)
        ax.set_ylabel('Probability', fontsize=14)
        ax.set_title(f'Deep Learning Prediction Probability: {seq_id}', fontsize=16)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=12)
        plt.tight_layout()
        img_path = os.path.join(self.output_dir, f"{fasta_basename}_{seq_id}_global_prob.png")
        plt.savefig(img_path, dpi=300)
        plt.close()
        return img_path
    
    @staticmethod
    def plot_placeholder(img_path, title):
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, title, ha='center', va='center', fontsize=18, color='gray')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()
    
    def plot_structure(self, seq, genes, ori_start, ori_end, save_path, full_region=False):
        if full_region:
            padding = 500
            window_start = max(0, ori_start - padding)
            window_end = min(len(seq), ori_end + padding)
            title = "oriC and flanking gene structure (full predicted region)"
        else:
            center = (ori_start + ori_end) // 2
            window_start = max(0, center - 10000)
            window_end = min(len(seq), center + 10000)
            title = "oriC and flanking 10kb gene structure"
        region_genes = [g for g in genes if g['end'] > window_start and g['start'] < window_end]
        fig, ax = plt.subplots(figsize=(16, 3), dpi=200)
        y_levels = [0, 1, 2, 3, 4, 5]
        gene_tracks = []
        for g in region_genes:
            placed = False
            for y in y_levels:
                overlap = False
                for other in gene_tracks:
                    if other['y'] == y and not (g['end'] < other['start'] or g['start'] > other['end']):
                        overlap = True
                        break
                if not overlap:
                    gene_tracks.append({'start': g['start'], 'end': g['end'], 'y': y,
                                        'gene': g['gene'], 'product': g['product'], 'strand': g['strand']})
                    placed = True
                    break
            if not placed:
                gene_tracks.append({'start': g['start'], 'end': g['end'], 'y': y_levels[-1],
                                    'gene': g['gene'], 'product': g['product'], 'strand': g['strand']})
        for gt in gene_tracks:
            color = '#1f77b4' if gt['strand'] == 1 else '#ff7f0e'
            ax.arrow(gt['start'], gt['y'], gt['end']-gt['start'], 0,
                     head_width=0.3, head_length=300, fc=color, ec=color,
                     length_includes_head=True, linewidth=2)
            ax.text((gt['start']+gt['end'])/2, gt['y']+0.4, gt['gene'],
                    ha='center', va='bottom', fontsize=10, rotation=45, color=color)
            ax.text((gt['start']+gt['end'])/2, gt['y']-0.3, f"{gt['end']-gt['start']}bp",
                    ha='center', va='top', fontsize=8, color='gray')
        ax.axvspan(ori_start, ori_end, color='red', alpha=0.2, label='Predicted oriC')
        ax.set_xlim(window_start, window_end)
        ax.set_ylim(-1, max(y_levels)+1)
        ax.set_xlabel('Genome Position', fontsize=14)
        ax.set_ylabel('Gene Layer', fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
    
    # --- Helper methods for recommend_oric ---
    def _find_intergenic_regions_in_range(self, search_start, search_end, genes, seq_length, circular):
        intergenic = []
        if not genes:
            return [{'start': search_start, 'end': search_end}]
        relevant = [g for g in genes if g['end'] > search_start and g['start'] < search_end]
        if not relevant:
            return [{'start': search_start, 'end': search_end}]
        relevant.sort(key=lambda x: x['start'])
        intergenic.append({'start': search_start, 'end': relevant[0]['start']})
        for i in range(len(relevant)-1):
            if relevant[i+1]['start'] > relevant[i]['end']:
                intergenic.append({'start': relevant[i]['end'], 'end': relevant[i+1]['start']})
        intergenic.append({'start': relevant[-1]['end'], 'end': search_end})
        intergenic = [r for r in intergenic if r['end'] > r['start']]
        return intergenic
    
    def _calculate_dnaa_box_score(self, region_boxes, region_length):
        count = len(region_boxes)
        score = min(0.5, count * 0.05)
        return score
    
    def _calculate_indicator_gene_score(self, region_start, region_end, indicator_genes, genes, circular, seq_length):
        if not indicator_genes:
            return 0.0, None, 1e9
        region_center = (region_start + region_end) // 2
        best_dist = 1e9
        best_gene = None
        for ig in indicator_genes:
            gene_center = (ig['start'] + ig['end']) // 2
            dist = calculate_circular_distance(region_center, gene_center, seq_length)
            if dist < best_dist:
                best_dist = dist
                best_gene = ig
        if best_dist <= 5000:
            score = 0.4 * (1 - best_dist/5000)
        else:
            score = 0.0
        return score, best_gene, best_dist
    
    def _calculate_gc_skew_score(self, region_center, gc_ori_pos, seq_length, circular):
        if gc_ori_pos is None:
            return 0.0
        dist = calculate_circular_distance(region_center, gc_ori_pos, seq_length)
        if dist <= 10000:
            score = 0.3 * (1 - dist/10000)
        else:
            score = 0.0
        return score
    
    def _calculate_priority_score(self, region_start, region_end, dl_peaks, genes, seq_length, circular):
        dl_score = 0.3
        intergenic_score = 0.2 if self._is_intergenic_region(region_start, region_end, genes) else 0.0
        return dl_score + intergenic_score
    
    def _count_gatc_motifs(self, sequence):
        if not sequence:
            return 0
        return sequence.upper().count('GATC')
    
    def _is_intergenic_region(self, start, end, genes):
        for g in genes:
            if g['start'] >= start and g['end'] <= end:
                return False
        return True
    
    def recommend_oric(self, dl_peaks, dl_probs, gc_ori_pos, indicator_genes, genes, seq_length, seq=None, all_dnaa_boxes=None):
        recommendations = []
        if not dl_peaks:
            return None
        search_window = 3000
        for i, (peak_pos, peak_prob) in enumerate(zip(dl_peaks, dl_probs)):
            search_start = max(0, peak_pos - search_window)
            search_end = min(seq_length, peak_pos + search_window)
            intergenic_regions = self._find_intergenic_regions_in_range(search_start, search_end, genes, seq_length, False)
            for intergenic_region in intergenic_regions:
                region_start = intergenic_region['start']
                region_end = intergenic_region['end']
                region_center = (region_start + region_end) // 2
                region_length = region_end - region_start
                region_boxes = []
                if all_dnaa_boxes:
                    for box in all_dnaa_boxes:
                        if region_start <= box['start'] < region_end:
                            region_boxes.append(box)
                dnaa_score = self._calculate_dnaa_box_score(region_boxes, region_length)
                if dnaa_score <= 0:
                    dnaa_score = 0.05
                indicator_score, closest_gene, distance_to_gene = self._calculate_indicator_gene_score(
                    region_start, region_end, indicator_genes, genes, False, seq_length
                )
                gc_score = self._calculate_gc_skew_score(region_center, gc_ori_pos, seq_length, False)
                priority_score = self._calculate_priority_score(region_start, region_end, dl_peaks, genes, seq_length, False)
                total_score = dnaa_score + indicator_score + gc_score + priority_score
                region_sequence = ""
                if seq and region_start < len(seq) and region_end <= len(seq):
                    region_sequence = str(seq[region_start:region_end])
                gatc_count = self._count_gatc_motifs(region_sequence)
                priority_level = "High" if priority_score >= 0.3 else "Medium" if priority_score >= 0.2 else "Low"
                recommendations.append({
                    'start': region_start,
                    'end': region_end,
                    'length': region_length,
                    'total_score': total_score,
                    'dl_probability': peak_prob,
                    'dnaa_box_count': len(region_boxes),
                    'closest_indicator_gene': closest_gene['gene'] if closest_gene else '',
                    'distance_to_gene': distance_to_gene,
                    'gc_skew_distance': calculate_circular_distance(region_center, gc_ori_pos, seq_length) if gc_ori_pos else seq_length,
                    'gatc_motif_count': gatc_count,
                    'is_intergenic': self._is_intergenic_region(region_start, region_end, genes),
                    'sequence': region_sequence,
                    'priority_level': priority_level,
                    'priority_score': priority_score,
                    'source': 'DL_region'
                })
        unique = {}
        for rec in recommendations:
            key = rec['start']
            if key not in unique or rec['total_score'] > unique[key]['total_score']:
                unique[key] = rec
        recommendations = sorted(unique.values(), key=lambda x: x['total_score'], reverse=True)
        return recommendations
    
    # --- Main prediction pipeline ---
    def predict(self, fasta_path, gene_file=None, indicator_gene_list=None):
        results = []
        gcskew_img_paths = {}
        seq_records = {}
        zcurve_img_paths = {}
        dnaa_img_paths = {}
        global_prob_img_paths = {}
        
        base = os.path.splitext(os.path.splitext(os.path.basename(fasta_path))[0])[0]
        genes = parse_gbk_genes(gene_file) if gene_file else []
        indicator_genes = []
        if indicator_gene_list and genes:
            indicator_names = set([name.strip().upper() for name in indicator_gene_list.split(',')])
            indicator_genes = [g for g in genes if g.get('gene', '').upper() in indicator_names]
        
        if fasta_path.endswith('.gz'):
            opener = gzip.open
        else:
            opener = open
        
        with opener(fasta_path, "rt") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq = str(record.seq)
                seq_id = record.id
                seq_records[seq_id] = seq
                
                window_starts = list(range(0, len(seq) - self.seq_length + 1, self.step_size))
                dl_positions = []
                dl_probs = []
                for i in window_starts:
                    window = seq[i:i + self.seq_length]
                    if len(window) == self.seq_length:
                        encoded = self.preprocess(window)
                        with torch.no_grad():
                            inputs = torch.tensor(encoded).unsqueeze(0).to(self.device)
                            output = self.model(inputs).squeeze()
                            prob = torch.sigmoid(output).cpu().item()
                        dl_positions.append(i)
                        dl_probs.append(prob)
                
                global_prob_img_paths[seq_id] = self.plot_global_probability(base, seq_id, dl_positions, dl_probs)
                dl_peaks, dl_peak_probs = self.find_peaks(dl_positions, dl_probs, threshold=self.threshold)
                
                gc_positions, gc_skew = self.calculate_gc_skew(seq)
                gcskew_img_path = self.plot_gc_skew_circular(base, seq_id, gc_positions, gc_skew, len(seq), dl_peaks if dl_peaks else None)
                gcskew_img_paths[seq_id] = gcskew_img_path
                
                dnaa_regions, dnaa_img = self.run_dnaa_box_analysis(seq, seq_id, base)
                dnaa_img_paths[seq_id] = dnaa_img
                _, zcurve_img = self.run_z_curve_analysis(seq, seq_id, base)
                zcurve_img_paths[seq_id] = zcurve_img
                
                if gc_skew:
                    min_idx = np.argmin(gc_skew)
                    gc_ori_pos = gc_positions[min_idx] if gc_positions else None
                else:
                    gc_ori_pos = None
                
                all_dnaa_boxes_sensitive = find_dnaa_boxes(seq, patterns=None, max_mismatches=3, include_reverse=True)
                recommendations = self.recommend_oric(dl_peaks, dl_peak_probs, gc_ori_pos, indicator_genes,
                                                      genes, len(seq), seq=seq, all_dnaa_boxes=all_dnaa_boxes_sensitive)
                best_recommendation = recommendations[0] if recommendations else None
                
                best_structure_img = None
                if best_recommendation:
                    best_structure_img = os.path.join(self.output_dir, f"{base}_{seq_id}_best_oric_structure.png")
                    self.plot_structure(seq, genes, best_recommendation['start'], best_recommendation['end'],
                                        best_structure_img, full_region=False)
                
                high_prob_regions = []
                for pos, prob in zip(dl_peaks, dl_peak_probs):
                    start = max(0, pos - self.seq_length//2)
                    end = min(len(seq), pos + self.seq_length//2)
                    region_seq = seq[start:end]
                    up, down = find_nearest_genes(start, end, genes)
                    structure_img = os.path.join(self.output_dir, f"{base}_{seq_id}_structure_{start}_{end}.png")
                    self.plot_structure(seq, genes, start, end, structure_img, full_region=False)
                    high_prob_regions.append({
                        'start': start, 'end': end, 'max_prob': prob, 'sequence': region_seq,
                        'upstream_gene': up, 'downstream_gene': down, 'structure_img': structure_img
                    })
                high_prob_regions = self.merge_overlapping_predictions(high_prob_regions, full_sequence=seq)
                for region in high_prob_regions:
                    up, down = find_nearest_genes(region['start'], region['end'], genes)
                    region['upstream_gene'] = up
                    region['downstream_gene'] = down
                    structure_img = os.path.join(self.output_dir, f"{base}_{seq_id}_structure_merged_{region['start']}_{region['end']}.png")
                    self.plot_structure(seq, genes, region['start'], region['end'], structure_img, full_region=False)
                    region['structure_img'] = structure_img
                    region['sequence'] = seq[region['start']:region['end']]
                
                results.append({
                    'id': seq_id,
                    'length': len(seq),
                    'regions': high_prob_regions,
                    'recommendations': recommendations,
                    'best_recommendation': best_recommendation,
                    'best_structure_img': best_structure_img
                })
                
                # Only generate flanking10kb_genes.csv, skip structure_genes.csv
                flanking_csv = os.path.join(self.output_dir, f"{base}_{seq_id}_flanking10kb_genes.csv")
                write_structure_genes_csv(high_prob_regions, genes, flanking_csv)
        
        self._extra_results = {'zcurve_img_paths': zcurve_img_paths, 'dnaa_img_paths': dnaa_img_paths,
                               'global_prob_img_paths': global_prob_img_paths}
        return results, gcskew_img_paths, seq_records
    
    def save_results(self, results, output_path):
        with open(output_path, "w") as f:
            f.write("SequenceID,Length,Region_Start,Region_End,Probability\n")
            for result in results:
                if not result['regions']:
                    f.write(f"{result['id']},{result['length']},-,-,0.0\n")
                else:
                    for region in result['regions']:
                        f.write(f"{result['id']},{result['length']},{region['start']},{region['end']},{region['max_prob']:.4f}\n")
    
    def save_html_report(self, results, gcskew_img_paths, seq_records, output_path):
        html = []
        html.append("<!DOCTYPE html><html><head><meta charset='UTF-8'><title>DNA Replication Origin Prediction Report</title><style>")
        html.append("body { font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }")
        html.append("h1, h2, h3, h4 { color: #2c3e50; }")
        html.append("table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }")
        html.append("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
        html.append("th { background-color: #f2f2f2; }")
        html.append("tr:nth-child(even) { background-color: #f9f9f9; }")
        html.append("pre { background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; }")
        html.append(".region-block { margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }")
        html.append(".gcskew-img { display: block; margin: 0 auto 20px auto; border: 1px solid #ccc; }")
        html.append(".seq-label { font-weight: bold; color: #333; }")
        html.append("</style></head><body>")
        html.append("<h1>DNA Replication Origin Prediction Report</h1>")
        
        for result in results:
            seq_id = result['id']
            html.append(f"<h2>Sequence: {seq_id} (Length: {result['length']})</h2>")
            
            if seq_id in gcskew_img_paths:
                with open(gcskew_img_paths[seq_id], "rb") as imgf:
                    img_b64 = base64.b64encode(imgf.read()).decode('utf-8')
                html.append("<h3>GC Skew Circular Plot</h3>")
                html.append(f"<img class='gcskew-img' src='data:image/png;base64,{img_b64}' width='400'>")
            
            if seq_id in self._extra_results.get('global_prob_img_paths', {}):
                with open(self._extra_results['global_prob_img_paths'][seq_id], "rb") as imgf:
                    img_b64 = base64.b64encode(imgf.read()).decode('utf-8')
                html.append("<h3>Global Deep Learning Prediction Probability</h3>")
                html.append(f"<img class='gcskew-img' src='data:image/png;base64,{img_b64}' width='900'>")
            
            if seq_id in self._extra_results.get('dnaa_img_paths', {}):
                with open(self._extra_results['dnaa_img_paths'][seq_id], "rb") as imgf:
                    img_b64 = base64.b64encode(imgf.read()).decode('utf-8')
                html.append("<h3>DnaA Box Analysis (max mismatches=1)</h3>")
                html.append(f"<img class='gcskew-img' src='data:image/png;base64,{img_b64}' width='600'>")
            
            if seq_id in self._extra_results.get('zcurve_img_paths', {}):
                with open(self._extra_results['zcurve_img_paths'][seq_id], "rb") as imgf:
                    img_b64 = base64.b64encode(imgf.read()).decode('utf-8')
                html.append("<h3>Z-curve Analysis</h3>")
                html.append(f"<img class='gcskew-img' src='data:image/png;base64,{img_b64}' width='600'>")
            
            # DeepBacOri Predicted Regions (merged) - TABLE
            html.append("<h3>DeepBacOri Predicted Regions (merged)</h3>")
            if not result['regions']:
                html.append("<p><em>No predicted origin regions found.</em></p>")
            else:
                html.append("<table><thead><tr><th>Rank</th><th>Start</th><th>End</th><th>Probability</th><th>Upstream Gene</th><th>Downstream Gene</th></tr></thead><tbody>")
                for i, region in enumerate(result['regions'], 1):
                    up = region.get('upstream_gene', {}).get('gene', '')
                    down = region.get('downstream_gene', {}).get('gene', '')
                    html.append(f"<tr><td>{i}</td><td>{region['start']}</td><td>{region['end']}</td><td>{region['max_prob']:.4f}</td><td>{up}</td><td>{down}</td></tr>")
                html.append("</tbody></table>")
                for i, region in enumerate(result['regions'], 1):
                    html.append(f"<div class='region-block'><span class='seq-label'>Region {i} Sequence:</span>")
                    html.append(f"<pre>{region['sequence']}</pre></div>")
                    if 'structure_img' in region and os.path.exists(region['structure_img']):
                        with open(region['structure_img'], "rb") as imgf:
                            img_b64 = base64.b64encode(imgf.read()).decode('utf-8')
                        html.append("<h4>oriC and flanking 10kb gene structure (merged region)</h4>")
                        html.append(f"<img class='gcskew-img' src='data:image/png;base64,{img_b64}' width='800'>")
        
        # Recommendation section (placed at the end)
        html.append("<hr><h1>oriC Candidate Recommendations</h1>")
        for result in results:
            if not result.get('recommendations'):
                continue
            seq_id = result['id']
            html.append(f"<h2>Recommendations for {seq_id}</h2>")
            best = result.get('best_recommendation')
            if best:
                html.append("<h3>Top Scoring Candidate (Best oriC)</h3>")
                html.append("<table><thead><tr><th>Start</th><th>End</th><th>Total Score</th><th>DL Probability</th><th>DnaA Boxes</th><th>Indicator Gene</th><th>Priority</th><th>GATC count</th></tr></thead><tbody>")
                html.append(f"<tr><td>{best['start']}</td><td>{best['end']}</td><td>{best['total_score']:.3f}</td><td>{best['dl_probability']:.3f}</td><td>{best['dnaa_box_count']}</td><td>{best['closest_indicator_gene']}</td><td>{best['priority_level']}</td><td>{best['gatc_motif_count']}</td></tr>")
                html.append("</tbody></table>")
                if result.get('best_structure_img') and os.path.exists(result['best_structure_img']):
                    with open(result['best_structure_img'], "rb") as imgf:
                        img_b64 = base64.b64encode(imgf.read()).decode('utf-8')
                    html.append("<h4>Gene Structure (10kb flanking) of Best Candidate</h4>")
                    html.append(f"<img class='gcskew-img' src='data:image/png;base64,{img_b64}' width='800'>")
                
                html.append("<h3>Top 5 Ranked Candidates</h3>")
                html.append("<table><thead><tr><th>Rank</th><th>Start</th><th>End</th><th>Total Score</th><th>DL Prob</th><th>DnaA Boxes</th><th>Indicator Gene</th><th>Priority</th></tr></thead><tbody>")
                for idx, rec in enumerate(result['recommendations'][:5], 1):
                    html.append(f"<tr><td>{idx}</td><td>{rec['start']}</td><td>{rec['end']}</td><td>{rec['total_score']:.3f}</td><td>{rec['dl_probability']:.3f}</td><td>{rec['dnaa_box_count']}</td><td>{rec['closest_indicator_gene']}</td><td>{rec['priority_level']}</td></tr>")
                html.append("</tbody></table>")
            else:
                html.append("<p>No recommendations generated.</p>")
        
        html.append("<hr><p><em>Generated by DeepBacOri DNA Replication Origin Predictor</em></p></body></html>")
        with open(output_path, 'w') as f:
            f.write('\n'.join(html))
        print(f"HTML report saved to {output_path}")

def parse_gbk_genes(gbk_path):
    genes = []
    if gbk_path is None:
        return genes
    if gbk_path.endswith('.gz'):
        opener = gzip.open
    else:
        opener = open
    with opener(gbk_path, "rt") as handle:
        for record in SeqIO.parse(handle, "genbank"):
            for feature in record.features:
                if feature.type == "CDS":
                    start = int(feature.location.start)
                    end = int(feature.location.end)
                    strand = int(feature.location.strand)
                    locus_tag = feature.qualifiers.get("locus_tag", [""])[0]
                    gene = feature.qualifiers.get("gene", [""])[0]
                    product = feature.qualifiers.get("product", [""])[0]
                    genes.append({"start": start, "end": end, "strand": strand,
                                  "locus_tag": locus_tag, "gene": gene, "product": product})
    return sorted(genes, key=lambda x: x['start'])

def find_nearest_genes(ori_start, ori_end, genes):
    upstream = None
    downstream = None
    for g in genes:
        if g['end'] <= ori_start:
            upstream = g
        elif g['start'] >= ori_end and downstream is None:
            downstream = g
            break
    return upstream, downstream

def write_structure_genes_csv(regions, genes, output_csv):
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["region_rank", "region_start", "region_end", "gene_start", "gene_end",
                         "gene", "product", "strand", "length"])
        for i, region in enumerate(regions, 1):
            center = (region['start'] + region['end']) // 2
            window_start = max(0, center - 10000)
            window_end = center + 10000
            for g in genes:
                if g['end'] < window_start or g['start'] > window_end:
                    continue
                writer.writerow([i, region['start'], region['end'], g['start'], g['end'],
                                 g['gene'], g['product'], g['strand'], g['end']-g['start']])

def process_one_genome(fasta_path, gene_file, output_dir, indicator_genes=None):
    base = os.path.splitext(os.path.splitext(os.path.basename(fasta_path))[0])[0]
    print(f"[Batch] Predicting for {fasta_path}")
    predictor = DNAPredictor(output_dir=output_dir)
    results, gcskew_img_paths, seq_records = predictor.predict(fasta_path, gene_file=gene_file, indicator_gene_list=indicator_genes)
    output_csv = os.path.join(output_dir, f"{base}_predictions.csv")
    output_html = os.path.join(output_dir, f"{base}_report.html")
    predictor.save_results(results, output_csv)
    predictor.save_html_report(results, gcskew_img_paths, seq_records, output_html)
    return f"{base} finished"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict DNA replication origin sites")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-i", "--input", help="Input FASTA file path (single genome)")
    group.add_argument("-d", "--input_dir", help="Input directory containing multiple FASTA files")
    parser.add_argument("-g", "--gene_file", help="Prodigal gbk gene file for single genome")
    parser.add_argument("-gd", "--gene_dir", help="Directory containing gbk gene files for batch mode")
    parser.add_argument("-o", "--output", help="Output directory for all results")
    parser.add_argument("--indicator_genes", help="Comma-separated list of indicator gene names (e.g., dnaA,gyrB)", default="dnaA,dnaN,mnmG")
    args = parser.parse_args()
    
    output_dir = args.output or "DeepBacOri_predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    if args.input_dir:
        import glob
        fasta_files = []
        for ext in ("*.fa", "*.fasta", "*.fna", "*.fna.gz", "*.ffn", "*.faa"):
            fasta_files.extend(glob.glob(os.path.join(args.input_dir, ext)))
        if not fasta_files:
            print(f"No FASTA files found in {args.input_dir}")
            exit(1)
        gene_file_map = {}
        if args.gene_dir:
            for fasta_path in fasta_files:
                base = os.path.splitext(os.path.splitext(os.path.basename(fasta_path))[0])[0]
                candidate = os.path.join(args.gene_dir, base + ".gbff.gz")
                if os.path.exists(candidate):
                    gene_file_map[fasta_path] = candidate
                else:
                    candidate = os.path.join(args.gene_dir, base + ".gbk")
                    gene_file_map[fasta_path] = candidate if os.path.exists(candidate) else None
        else:
            for fasta_path in fasta_files:
                gene_file_map[fasta_path] = None
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = []
            for fasta_path in fasta_files:
                gene_file = gene_file_map[fasta_path]
                futures.append(executor.submit(process_one_genome, fasta_path, gene_file, output_dir, args.indicator_genes))
            for fut in as_completed(futures):
                print(fut.result())
    else:
        base = os.path.splitext(os.path.splitext(os.path.basename(args.input))[0])[0]
        gene_file = args.gene_file
        predictor = DNAPredictor(output_dir=output_dir)
        results, gcskew_img_paths, seq_records = predictor.predict(args.input, gene_file=gene_file, indicator_gene_list=args.indicator_genes)
        output_csv = os.path.join(output_dir, f"{base}_predictions.csv")
        output_html = os.path.join(output_dir, f"{base}_report.html")
        predictor.save_results(results, output_csv)
        predictor.save_html_report(results, gcskew_img_paths, seq_records, output_html)
        print(f"[Single] Results saved to {output_csv}, HTML report saved to {output_html}")
