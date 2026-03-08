#!/usr/bin/env python3
"""
Evaluate watermark detection performance by calculating FPR, TPR, TNR, FNR.
"""

import argparse
import json
import os
import re
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_strength_from_filename(filename: str) -> float:
    """Extract strength value from filename."""
    match = re.search(r'strength_([0-9.]+)', filename)
    if match:
        return float(match.group(1))
    return None


def extract_fraction_from_filename(filename: str) -> Optional[float]:
    """Extract fraction value from filename."""
    match = re.search(r'frac_([0-9.]+)', filename)
    if match:
        return float(match.group(1))
    return None


def has_only_english_flag(filename: str) -> bool:
    """Return True if filename includes only_English marker."""
    return "only_English" in filename


def load_z_scores(filepath: str) -> List[float]:
    """Load z_scores from a JSONL file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data.get('z_score', [])


def load_z_file_data(filepath: Path) -> Tuple[List[float], Optional[float]]:
    """Load z_scores and avg_ppl from a _z.jsonl file.
    
    Returns:
        Tuple of (z_scores list, avg_ppl value or None if not present)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
        z_scores = data.get('z_score', [])
        avg_ppl = data.get('avg_ppl', None)
        return z_scores, avg_ppl


def load_z_file_data_with_labels(filepath: Path) -> Tuple[List[float], Optional[float], Optional[int], Optional[int]]:
    """Load z_scores, avg_ppl, positive_num, and negative_num from a _z.jsonl file.
    
    Returns:
        Tuple of (z_scores list, avg_ppl value or None, positive_num or None, negative_num or None)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
        z_scores = data.get('z_score', [])
        avg_ppl = data.get('avg_ppl', None)
        positive_num = data.get('positive_num', None)
        negative_num = data.get('negative_num', None)
        return z_scores, avg_ppl, positive_num, negative_num


def calculate_fpr(z_scores: List[float], tau: float) -> float:
    """Calculate False Positive Rate: fraction of z_scores >= tau."""
    if len(z_scores) == 0:
        return 0.0
    count = sum(1 for z in z_scores if z >= tau)
    return count / len(z_scores)


def calculate_tpr(z_scores: List[float], tau: float) -> float:
    """Calculate True Positive Rate: fraction of z_scores >= tau."""
    if len(z_scores) == 0:
        return 0.0
    count = sum(1 for z in z_scores if z >= tau)
    return count / len(z_scores)


def calculate_auc_roc(z_scores: List[float], labels: List[int]) -> float:
    """Calculate AUC-ROC from z_scores and binary labels using the trapezoidal rule.

    Args:
        z_scores: List of z-scores (higher = more likely positive/watermarked)
        labels: List of true labels (1 for positive, 0 for negative)

    Returns:
        AUC-ROC value in [0, 1], or 0.0 if degenerate input
    """
    if len(z_scores) == 0 or len(z_scores) != len(labels):
        return 0.0

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Sort by z_score descending; ties are handled by accumulating before updating AUC
    sorted_pairs = sorted(zip(z_scores, labels), key=lambda x: x[0], reverse=True)

    auc = 0.0
    tp, fp = 0, 0
    tpr_prev, fpr_prev = 0.0, 0.0
    prev_score = None

    for score, label in sorted_pairs:
        if prev_score is not None and score != prev_score:
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    # Add the final segment to (1.0, 1.0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2

    return auc


def compute_roc_curve(z_scores: List[float], labels: List[int]) -> Tuple[List[float], List[float]]:
    """Compute ROC curve (fpr_points, tpr_points) by sweeping all unique thresholds.

    Returns lists starting at (0, 0) and ending at (1, 1).
    """
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return [0.0, 1.0], [0.0, 1.0]

    sorted_pairs = sorted(zip(z_scores, labels), key=lambda x: x[0], reverse=True)

    fpr_pts = [0.0]
    tpr_pts = [0.0]
    tp, fp = 0, 0
    prev_score = None

    for score, label in sorted_pairs:
        if prev_score is not None and score != prev_score:
            fpr_pts.append(fp / n_neg)
            tpr_pts.append(tp / n_pos)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score

    fpr_pts.append(fp / n_neg)
    tpr_pts.append(tp / n_pos)

    return fpr_pts, tpr_pts


def plot_roc_curves(
    roc_groups: Dict,
    group_key_label: str,
    output_png: Path,
    model_name: str,
) -> None:
    """Plot ROC curves grouped by only_English and save as PNG.

    Args:
        roc_groups: {only_english: [{'label': str, 'fpr': list, 'tpr': list, 'auc': float}, ...]}
        group_key_label: human-readable name for the curve identifier (e.g. 'Fraction' or 'Strength')
        output_png: destination path for the PNG file
        model_name: used in the figure title
    """
    n_groups = len(roc_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5), squeeze=False)

    colors = plt.cm.tab10.colors

    for col_idx, only_english in enumerate(sorted(roc_groups.keys())):
        ax = axes[0][col_idx]
        curves = roc_groups[only_english]

        for i, curve in enumerate(curves):
            ax.plot(
                curve['fpr'], curve['tpr'],
                color=colors[i % len(colors)],
                label=f"{curve['label']} (AUC={curve['auc']:.3f})",
                linewidth=1.5,
            )

        ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, label='Random')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        suffix = ' (only_English)' if only_english else ''
        ax.set_title(f"{model_name}{suffix}\nROC by {group_key_label}")
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC plot saved to {output_png}")


def calculate_metrics_from_labels(z_scores: List[float], labels: List[int], tau: float) -> Tuple[float, float, float, float]:
    """Calculate FPR, TPR, TNR, FNR from z_scores and true labels.
    
    Args:
        z_scores: List of z-scores (predictions)
        labels: List of true labels (1 for positive/watermark, 0 for negative/no watermark)
        tau: Threshold for prediction
    
    Returns:
        Tuple of (FPR, TPR, TNR, FNR)
    """
    if len(z_scores) != len(labels):
        raise ValueError(f"z_scores length ({len(z_scores)}) != labels length ({len(labels)})")
    
    if len(z_scores) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Predictions: 1 if z_score >= tau, 0 otherwise
    predictions = [1 if z >= tau else 0 for z in z_scores]
    
    # Calculate confusion matrix
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)  # True Positive
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)  # False Positive
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)  # True Negative
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)  # False Negative
    
    # Calculate metrics
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate (Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    
    return fpr, tpr, tnr, fnr


def find_threshold(z_scores: List[float], target_fpr: float = 0.01) -> float:
    """
    Find threshold tau from tau_list (0.0 to 15.0, step=0.5) 
    such that FPR < target_fpr.
    """
    tau_list = [0.0 + i * 0.5 for i in range(int((15.0 - 0.0) / 0.5) + 1)]
    for tau in tau_list:
        fpr = calculate_fpr(z_scores, tau)
        if fpr < target_fpr:
            return tau
    
    # If no threshold found, return the last one
    print(f"No threshold found!")
    exit(1)
    # return tau_list[-1]


def process_directory_fraction(directory: str, output_csv: str = None, tau_thres: float = None, target_fpr: float = 0.1):
    """
    Process all _z.jsonl files in the directory and calculate metrics using fraction mode.
    In fraction mode, we use positive_num and negative_num from JSON files to construct true labels.
    When tau_thres is not provided, each fraction file gets its own tau by finding threshold
    on that file's negative z_scores such that FPR < target_fpr (FPR@target_fpr per file).

    Args:
        directory: Path to directory containing _z.jsonl files
        output_csv: Path to output CSV file (default: same directory as input)
        tau_thres: If provided, use this threshold for all files in each group; otherwise find tau per file
        target_fpr: Target FPR for find_threshold when tau_thres is None (used per file)
    """
    directory = Path(directory)
    
    # Extract model name from directory name
    model_name = directory.name
    
    # Find all _z.jsonl files
    z_files = list(directory.glob('*_z.jsonl'))
    
    if not z_files:
        print(f"No _z.jsonl files found in {directory}")
        return
    
    # Separate files by (fraction, only_English) tuple
    fraction_files = {}
    for filepath in z_files:
        fraction = extract_fraction_from_filename(filepath.name)
        if fraction is not None:
            only_english = has_only_english_flag(filepath.name)
            key = (fraction, only_english)
            fraction_files[key] = filepath
    
    # Group files by only_English flag
    only_english_groups = {}
    for (fraction, only_english), filepath in fraction_files.items():
        if only_english not in only_english_groups:
            only_english_groups[only_english] = {}
        only_english_groups[only_english][fraction] = filepath
    
    # Prepare results and ROC data
    results = []
    roc_groups: Dict[bool, List[Dict]] = {}

    # Process each only_English group separately
    for only_english in sorted(only_english_groups.keys()):
        group_files = only_english_groups[only_english]
        roc_groups[only_english] = []

        print(f"\nProcessing only_English={only_english} group...")
        if tau_thres is not None:
            print(f"Using provided threshold tau = {tau_thres} for all fractions in this group")
        
        # Process every fraction file (each gets its own tau when tau_thres is None)
        for fraction in sorted(group_files.keys()):
            print(f"Processing fraction={fraction} (only_English={only_english})...")
            z_scores, avg_ppl, positive_num, negative_num = load_z_file_data_with_labels(group_files[fraction])
            print(f"Loaded {len(z_scores)} z_scores from fraction={fraction} file")
            if avg_ppl is not None:
                print(f"  avg_ppl: {avg_ppl:.4f}")
            
            if positive_num is None or negative_num is None:
                print(f"Warning: positive_num or negative_num not found in file {group_files[fraction]}, skipping...")
                continue
            
            print(f"  positive_num: {positive_num}, negative_num: {negative_num}")
            
            if len(z_scores) != positive_num + negative_num:
                print(f"Warning: z_scores length ({len(z_scores)}) != positive_num + negative_num ({positive_num + negative_num}), skipping...")
                continue
            
            labels = [1] * positive_num + [0] * negative_num
            negative_z_scores = z_scores[positive_num:]
            
            if tau_thres is not None:
                tau = tau_thres
            else:
                if negative_num == 0:
                    print(f"Warning: no negative samples in fraction={fraction}, cannot find tau, skipping...")
                    continue
                print(f"  Finding tau for this file (FPR@target_fpr={target_fpr})...")
                tau = find_threshold(negative_z_scores, target_fpr=target_fpr)
                print(f"  Found tau = {tau}")
            
            fpr, tpr, tnr, fnr = calculate_metrics_from_labels(z_scores, labels, tau)
            auc_roc = calculate_auc_roc(z_scores, labels)
            fpr_pts, tpr_pts = compute_roc_curve(z_scores, labels)

            results.append({
                'model_name': model_name,
                'fraction': fraction,
                'only_English': only_english,
                'length': len(z_scores),
                'tau': tau,
                'FPR': fpr,
                'TNR': tnr,
                'TPR': tpr,
                'FNR': fnr,
                'AUC_ROC': auc_roc,
                'avg_ppl': avg_ppl if avg_ppl is not None else ''
            })
            roc_groups[only_english].append({
                'label': f'Fraction {fraction}',
                'fpr': fpr_pts,
                'tpr': tpr_pts,
                'auc': auc_roc,
            })

    # Write to CSV
    if output_csv is None:
        output_csv = directory / f"{model_name}_evaluation.csv"
    else:
        output_csv = Path(output_csv)
    
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['model_name', 'fraction', 'only_English', 'length', 'tau', 'FPR', 'TNR', 'TPR', 'FNR', 'AUC_ROC', 'avg_ppl']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {output_csv}")

    # Plot and save ROC curves
    output_png = Path(str(output_csv).replace('.csv', '.png'))
    non_empty_roc = {k: v for k, v in roc_groups.items() if v}
    if non_empty_roc:
        plot_roc_curves(non_empty_roc, 'Fraction', output_png, model_name)

    print("="*100)
    print(f"\nSummary:")
    # Group results by only_English for summary
    by_only_english = {}
    for result in results:
        oe = result['only_English']
        if oe not in by_only_english:
            by_only_english[oe] = []
        by_only_english[oe].append(result)
    
    for only_english in sorted(by_only_english.keys()):
        group_results = by_only_english[only_english]
        print(f"\n  only_English={only_english}:")
        for result in group_results:
            print(f"    Fraction {result['fraction']}: tau={result['tau']}, FPR={result['FPR']:.4f}, TPR={result['TPR']:.4f}, TNR={result['TNR']:.4f}, FNR={result['FNR']:.4f}, AUC_ROC={result['AUC_ROC']:.4f}")
    print("="*100)


def process_directory(directory: str, output_csv: str = None, tau_thres: float = None):
    """
    Process all _z.jsonl files in the directory and calculate metrics.
    
    Args:
        directory: Path to directory containing _z.jsonl files
        output_csv: Path to output CSV file (default: same directory as input)
        tau_thres: If provided, use this threshold directly; otherwise find via find_threshold()
    """
    directory = Path(directory)
    
    # Extract model name from directory name
    model_name = directory.name
    
    # Find all _z.jsonl files
    z_files = list(directory.glob('*_z.jsonl'))
    
    if not z_files:
        print(f"No _z.jsonl files found in {directory}")
        return
    
    # Separate files by (strength, only_English) tuple
    strength_files = {}
    for filepath in z_files:
        strength = extract_strength_from_filename(filepath.name)
        if strength is not None:
            only_english = has_only_english_flag(filepath.name)
            key = (strength, only_english)
            strength_files[key] = filepath
    
    # Group files by only_English flag
    only_english_groups = {}
    for (strength, only_english), filepath in strength_files.items():
        if only_english not in only_english_groups:
            only_english_groups[only_english] = {}
        only_english_groups[only_english][strength] = filepath
    
    # Prepare results and ROC data
    results = []
    roc_groups: Dict[bool, List[Dict]] = {}

    # Process each only_English group separately
    for only_english in sorted(only_english_groups.keys()):
        group_files = only_english_groups[only_english]
        roc_groups[only_english] = []

        # Check if strength 0.0 file exists for this group
        if 0.0 not in group_files:
            print(f"No strength=0.0 file found for only_English={only_english} in {directory}")
            continue
        
        print(f"\nProcessing only_English={only_english} group...")
        
        # Load z_scores and avg_ppl from strength=0.0 file for this group
        print(f"Loading z_scores from strength=0.0 file (only_English={only_english})...")
        z_scores_0, avg_ppl_0 = load_z_file_data(group_files[0.0])
        print(f"Loaded {len(z_scores_0)} z_scores from strength=0.0 file")
        if avg_ppl_0 is not None:
            print(f"  avg_ppl: {avg_ppl_0:.4f}")
        
        # Find or use threshold tau for this group
        if tau_thres is not None:
            tau = tau_thres
            print(f"Using provided threshold tau = {tau}")
        else:
            print("Finding threshold tau...")
            tau = find_threshold(z_scores_0, target_fpr=0.01)
            print(f"Found threshold tau = {tau}")
        
        # Calculate FPR for strength=0.0
        fpr = calculate_fpr(z_scores_0, tau)
        tnr = 1 - fpr  # True Negative Rate
        
        # Add result for strength=0.0
        results.append({
            'model_name': model_name,
            'strength': 0.0,
            'only_English': only_english,
            'length': len(z_scores_0),
            'tau': tau,
            'FPR': fpr,
            'TNR': tnr,
            'TPR': 0.0,  # TPR is 0 for strength=0.0 (no watermark)
            'FNR': 1.0,  # FNR is 1.0 for strength=0.0
            'AUC_ROC': '',
            'avg_ppl': avg_ppl_0 if avg_ppl_0 is not None else ''
        })
        
        # Process other strength files in this group
        other_strengths = sorted([s for s in group_files.keys() if s != 0.0])
        
        for strength in other_strengths:
            print(f"Processing strength={strength} (only_English={only_english})...")
            z_scores, avg_ppl = load_z_file_data(group_files[strength])
            print(f"Loaded {len(z_scores)} z_scores from strength={strength} file")
            if avg_ppl is not None:
                print(f"  avg_ppl: {avg_ppl:.4f}")
            
            tpr = calculate_tpr(z_scores, tau)
            fnr = 1 - tpr  # False Negative Rate

            # AUC-ROC: negatives = strength=0.0 scores, positives = this strength's scores
            combined_z = z_scores_0 + z_scores
            combined_labels = [0] * len(z_scores_0) + [1] * len(z_scores)
            auc_roc = calculate_auc_roc(combined_z, combined_labels)
            fpr_pts, tpr_pts = compute_roc_curve(combined_z, combined_labels)

            results.append({
                'model_name': model_name,
                'strength': strength,
                'only_English': only_english,
                'length': len(z_scores),
                'tau': tau,
                'FPR': fpr,  # Same FPR for all in this group (based on strength=0.0)
                'TNR': tnr,  # Same TNR for all in this group
                'TPR': tpr,
                'FNR': fnr,
                'AUC_ROC': auc_roc,
                'avg_ppl': avg_ppl if avg_ppl is not None else ''
            })
            roc_groups[only_english].append({
                'label': f'Strength {strength}',
                'fpr': fpr_pts,
                'tpr': tpr_pts,
                'auc': auc_roc,
            })

    # Write to CSV
    if output_csv is None:
        output_csv = directory / f"{model_name}_evaluation.csv"
    else:
        output_csv = Path(output_csv)
    
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['model_name', 'strength', 'only_English', 'length', 'tau', 'FPR', 'TNR', 'TPR', 'FNR', 'AUC_ROC', 'avg_ppl']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {output_csv}")

    # Plot and save ROC curves
    output_png = Path(str(output_csv).replace('.csv', '.png'))
    non_empty_roc = {k: v for k, v in roc_groups.items() if v}
    if non_empty_roc:
        plot_roc_curves(non_empty_roc, 'Strength', output_png, model_name)

    print("="*100)
    print(f"\nSummary:")
    # Group results by only_English for summary
    by_only_english = {}
    for result in results:
        oe = result['only_English']
        if oe not in by_only_english:
            by_only_english[oe] = []
        by_only_english[oe].append(result)
    
    for only_english in sorted(by_only_english.keys()):
        group_results = by_only_english[only_english]
        # Find strength=0.0 result for this group
        strength_0_result = next((r for r in group_results if r['strength'] == 0.0), None)
        if strength_0_result:
            print(f"\n  only_English={only_english}:")
            print(f"    Threshold tau: {strength_0_result['tau']}")
            print(f"    FPR (from strength=0.0): {strength_0_result['FPR']:.4f}")
            print(f"    TNR: {strength_0_result['TNR']:.4f}")
            for result in group_results:
                if result['strength'] != 0.0:
                    auc_str = f", AUC_ROC={result['AUC_ROC']:.4f}" if result['AUC_ROC'] != '' else ''
                    print(f"    Strength {result['strength']}: TPR={result['TPR']:.4f}, FNR={result['FNR']:.4f}{auc_str}")
    print("="*100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate watermark detection performance by calculating FPR, TPR, TNR, FNR.'
    )
    parser.add_argument(
        'directory',
        help='Path to directory containing _z.jsonl files'
    )

    parser.add_argument(
        '--tau_thres',
        type=float,
        default=None,
        metavar='TAU',
        help='Use this threshold directly instead of finding via find_threshold(); all only_English groups use the same tau'
    )
    parser.add_argument(
        '--fraction_or_strength',
        type=str,
        choices=['fraction', 'strength'],
        default='strength',
        help='Evaluation mode: "strength" uses original logic, "fraction" uses positive_num/negative_num from JSON files'
    )
    parser.add_argument(
        '--target_fpr',
        type=float,
        default=0.1,
        metavar='FPR',
        help='Target FPR for per-file threshold finding in fraction mode (default: 0.1)'
    )
    args = parser.parse_args()

    output_csv = os.path.join(args.directory, "evaluation.csv")

    if args.fraction_or_strength == 'fraction':
        process_directory_fraction(args.directory, output_csv, args.tau_thres, args.target_fpr)
    else:
        process_directory(args.directory, output_csv, args.tau_thres)
