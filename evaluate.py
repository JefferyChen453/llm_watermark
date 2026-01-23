#!/usr/bin/env python3
"""
Evaluate watermark detection performance by calculating FPR, TPR, TNR, FNR.
"""

import json
import os
import re
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Optional


def extract_strength_from_filename(filename: str) -> float:
    """Extract strength value from filename."""
    match = re.search(r'strength_([0-9.]+)', filename)
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


def find_threshold(z_scores: List[float], target_fpr: float = 0.1) -> float:
    """
    Find threshold tau from tau_list (7.0 to 0.0, step=-0.5) 
    such that FPR < target_fpr.
    """
    tau_list = [0.0 + i * 0.5 for i in range(int((7.0 - 0.0) / 0.5) + 1)]
    for tau in tau_list:
        fpr = calculate_fpr(z_scores, tau)
        if fpr < target_fpr:
            return tau
    
    # If no threshold found, return the last one
    print(f"No threshold found!")
    exit(1)
    # return tau_list[-1]


def process_directory(directory: str, output_csv: str = None):
    """
    Process all _z.jsonl files in the directory and calculate metrics.
    
    Args:
        directory: Path to directory containing _z.jsonl files
        output_csv: Path to output CSV file (default: same directory as input)
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
    
    # Prepare results
    results = []
    
    # Process each only_English group separately
    for only_english in sorted(only_english_groups.keys()):
        group_files = only_english_groups[only_english]
        
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
        
        # Find threshold tau for this group
        print("Finding threshold tau...")
        tau = find_threshold(z_scores_0, target_fpr=0.1)
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
                'avg_ppl': avg_ppl if avg_ppl is not None else ''
            })
    
    # Write to CSV
    if output_csv is None:
        output_csv = directory / f"{model_name}_evaluation.csv"
    else:
        output_csv = Path(output_csv)
    
    print(f"\nWriting results to {output_csv}...")
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['model_name', 'strength', 'only_English', 'length', 'tau', 'FPR', 'TNR', 'TPR', 'FNR', 'avg_ppl']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results written to {output_csv}")
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
                    print(f"    Strength {result['strength']}: TPR={result['TPR']:.4f}, FNR={result['FNR']:.4f}")
    print("="*100)


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <directory> [output_csv]")
        print("Example: python evaluate.py /path/to/directory")
        sys.exit(1)
    
    directory = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_directory(directory, output_csv)
