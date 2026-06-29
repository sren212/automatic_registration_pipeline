#!/usr/bin/env python3
"""
Evaluate rotation search accuracy across synthetically rotated datasets.
"""

import subprocess
import re
from pathlib import Path

def run_registration(omics_path: Path, output_dir: Path) -> float | None:
    """Run registration pipeline and extract detected rotation."""
    cmd = [
        "python", "registration_pipeline.py",
        "--atlas", "data/aba_nissl.nrrd",
        "--omics", str(omics_path),
        "--slice", "64",
        "--search-rotation",
        "--rotation-step", "1",
        "--output", str(output_dir)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Extract rotation from output
    match = re.search(r"Rotation used: ([\d.]+)°", result.stdout)
    if match:
        return float(match.group(1))
    return None

def extract_expected_rotation(filename: str) -> float:
    """Extract expected rotation from filename like 'xxx_rot30.csv'."""
    match = re.search(r"_rot(\d+)\.csv$", filename)
    if match:
        return float(match.group(1))
    return 0.0  # Original file has 0 rotation

def main():
    data_dir = Path("data")
    results_dir = Path("rotation_eval_results")
    results_dir.mkdir(exist_ok=True)
    
    # Find all rotated files
    files = sorted(data_dir.glob("abca3_slice_13_cell_metadata_rot*.csv"))
    
    # Include original file (0 rotation)
    original = data_dir / "abca3_slice_13_cell_metadata.csv"
    if original.exists():
        files = [original] + list(files)
    
    results = []
    
    for omics_path in files:
        expected = extract_expected_rotation(omics_path.name)
        output_dir = results_dir / f"rot{int(expected)}"
        
        print(f"\n{'='*60}")
        print(f"Processing: {omics_path.name}")
        print(f"Expected rotation: {expected}°")
        
        detected = run_registration(omics_path, output_dir)
        
        if detected is not None:
            # Handle wraparound (e.g., 359° vs 1° are close)
            error = abs(detected - expected)
            error = min(error, 360 - error)
            
            results.append({
                'file': omics_path.name,
                'expected': expected,
                'detected': detected,
                'error': error
            })
            
            print(f"Detected rotation: {detected}°")
            print(f"Error: {error}°")
        else:
            print("ERROR: Could not extract rotation from output")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'File':<45} {'Expected':>8} {'Detected':>8} {'Error':>8}")
    print("-" * 70)
    
    total_error = 0
    for r in results:
        print(f"{r['file']:<45} {r['expected']:>8.1f} {r['detected']:>8.1f} {r['error']:>8.1f}")
        total_error += r['error']
    
    if results:
        mean_error = total_error / len(results)
        max_error = max(r['error'] for r in results)
        print("-" * 70)
        print(f"Mean error: {mean_error:.2f}°")
        print(f"Max error: {max_error:.2f}°")
        print(f"Perfect matches (error < 1°): {sum(1 for r in results if r['error'] < 1)}/{len(results)}")
    
    # Save results to CSV
    import csv
    with open(results_dir / "rotation_accuracy.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'expected', 'detected', 'error'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {results_dir / 'rotation_accuracy.csv'}")

if __name__ == "__main__":
    main()