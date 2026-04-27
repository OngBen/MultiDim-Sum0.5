import os
import numpy as np
from collections import Counter, defaultdict
import re

# Import the necessary functions from your existing code
from utils_now import loadVocabulary, da_vocab_per_dim, dimensions, da_functions

def analyze_sampling_statistics(data_path, vocab_path, output_file):
    """
    Analyze and print oversampling/undersampling statistics with dimension-level aggregation
    """
    # Load vocabulary
    in_vocab = loadVocabulary(os.path.join(vocab_path, 'in_vocab'))
    
    # Initialize counters
    original_label_counts = Counter()
    final_label_counts = Counter()
    original_dimension_counts = Counter()
    final_dimension_counts = Counter()
    dialogue_counts = []
    
    # Read and process the training data
    in_path = os.path.join(data_path, 'train', 'in')
    da_path = os.path.join(data_path, 'train', 'da_iso_improved_3')
    
    with open(in_path, 'r') as f_in, open(da_path, 'r') as f_da:
        in_lines = f_in.readlines()
        da_lines = f_da.readlines()
    
    total_dialogues_before = len(da_lines)
    
    print(f"Processing {total_dialogues_before} dialogues...")
    
    # Count original label frequencies
    for da_line in da_lines:
        # Parse dimension-function pairs
        blocks = re.findall(r'\{([^}]+)\}', da_line.strip())
        dimension_functions = set()
        
        for block in blocks:
            items = re.split(r'\s*,\s*', block)
            for item in items:
                if ':' in item:
                    d, f = item.split(':', 1)
                    label = f"{d.strip()}:{f.strip()}"
                    dimension_functions.add(label)
                    original_label_counts[label] += 1
                    # Count by dimension
                    original_dimension_counts[d.strip()] += 1
        
        dialogue_counts.append({
            'input': in_lines[len(dialogue_counts)] if len(dialogue_counts) < len(in_lines) else "",
            'da': da_line.strip(),
            'dimension_functions': dimension_functions
        })
    
    # Calculate sampling weights (same logic as in _precompute_all_data)
    total_utterances = sum(len(re.findall(r'\{([^}]+)\}', data['da'])) for data in dialogue_counts)
    dimension_function_frequencies = Counter()
    
    for data in dialogue_counts:
        for df in data['dimension_functions']:
            dimension_function_frequencies[df] += 1
    
    sampling_weights = []
    for i, data in enumerate(dialogue_counts):
        weight = 0.0
        for df in data['dimension_functions']:
            frequency = dimension_function_frequencies[df] / total_utterances
            inverse_freq = 1.0 / (frequency + 1e-8)
            weight += np.sqrt(inverse_freq)
        
        if data['dimension_functions']:
            weight = weight / np.sqrt(len(data['dimension_functions']))
        else:
            weight = 1.0
        
        sampling_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(sampling_weights)
    sampling_weights = [w / total_weight for w in sampling_weights]
    
    # Simulate one epoch of sampling to get final counts
    total_dialogues_after = total_dialogues_before  # Same number of samples per epoch
    sampled_indices = np.random.choice(
        len(dialogue_counts), 
        size=total_dialogues_after, 
        p=sampling_weights,
        replace=True
    )
    
    # Count labels in the sampled dataset
    for idx in sampled_indices:
        data = dialogue_counts[idx]
        for df in data['dimension_functions']:
            final_label_counts[df] += 1
            # Count by dimension for final dataset
            d, f = df.split(':', 1)
            final_dimension_counts[d.strip()] += 1
    
    # Calculate changes per label
    label_changes = {}
    all_labels = set(original_label_counts.keys()) | set(final_label_counts.keys())
    
    for label in all_labels:
        original = original_label_counts.get(label, 0)
        final = final_label_counts.get(label, 0)
        change = final - original
        change_percent = (change / original * 100) if original > 0 else float('inf')
        label_changes[label] = {
            'original': original,
            'final': final,
            'change': change,
            'change_percent': change_percent
        }
    
    # Calculate changes per dimension
    dimension_changes = {}
    all_dimensions = set(original_dimension_counts.keys()) | set(final_dimension_counts.keys())
    
    for dimension in all_dimensions:
        original = original_dimension_counts.get(dimension, 0)
        final = final_dimension_counts.get(dimension, 0)
        change = final - original
        change_percent = (change / original * 100) if original > 0 else float('inf')
        dimension_changes[dimension] = {
            'original': original,
            'final': final,
            'change': change,
            'change_percent': change_percent
        }
    
    # Write results to file
    with open(output_file, 'w') as f:
        f.write("SAMPLING STATISTICS ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. DATASET COUNTS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total dialogues before sampling: {total_dialogues_before}\n")
        f.write(f"Total dialogues after sampling: {total_dialogues_after}\n")
        f.write(f"Sampling method: Weighted sampling with replacement\n")
        f.write(f"Total unique dimension-function labels: {len(all_labels)}\n\n")
        
        f.write("2. DIMENSION-LEVEL STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'Dimension':<30} {'Original':<10} {'Final':<10} {'Change':<10} {'Change %':<15}\n")
        f.write("-" * 75 + "\n")
        
        # Sort dimensions by name for consistency
        sorted_dimensions = sorted(dimension_changes.items(), key=lambda x: x[0])
        
        for dimension, stats in sorted_dimensions:
            change_str = f"+{stats['change']}" if stats['change'] > 0 else str(stats['change'])
            percent_str = f"{stats['change_percent']:+.1f}%" if stats['change_percent'] != float('inf') else "N/A"
            
            f.write(f"{dimension:<30} {stats['original']:<10} {stats['final']:<10} {change_str:<10} {percent_str:<15}\n")
        
        # Calculate dimension totals
        total_original_dim = sum(original_dimension_counts.values())
        total_final_dim = sum(final_dimension_counts.values())
        total_change_dim = total_final_dim - total_original_dim
        total_change_percent_dim = (total_change_dim / total_original_dim * 100) if total_original_dim > 0 else 0
        
        f.write(f"{'TOTAL':<30} {total_original_dim:<10} {total_final_dim:<10} {total_change_dim:+<10} {total_change_percent_dim:+.1f}%\n\n")
        
        f.write("3. LABEL DISTRIBUTION CHANGES (Detailed)\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'Label':<50} {'Original':<10} {'Final':<10} {'Change':<10} {'Change %':<15}\n")
        f.write("-" * 100 + "\n")
        
        # Sort by absolute change percentage
        sorted_labels = sorted(label_changes.items(), 
                             key=lambda x: abs(x[1]['change_percent']), 
                             reverse=True)
        
        for label, stats in sorted_labels:
            if stats['original'] > 0:  # Only show labels that existed originally
                change_str = f"+{stats['change']}" if stats['change'] > 0 else str(stats['change'])
                percent_str = f"{stats['change_percent']:+.1f}%" if stats['change_percent'] != float('inf') else "N/A"
                
                f.write(f"{label:<50} {stats['original']:<10} {stats['final']:<10} {change_str:<10} {percent_str:<15}\n")
        
        f.write("\n4. SAMPLING WEIGHT STATISTICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Min sampling weight: {min(sampling_weights):.6f}\n")
        f.write(f"Max sampling weight: {max(sampling_weights):.6f}\n")
        f.write(f"Average sampling weight: {np.mean(sampling_weights):.6f}\n")
        f.write(f"Uniform weight (1/N): {1/len(sampling_weights):.6f}\n")
        f.write(f"Weight ratio (max/min): {max(sampling_weights)/min(sampling_weights):.1f}x\n")
        
        # Count dialogues with significant weight changes
        uniform_weight = 1.0 / len(dialogue_counts)
        oversampled = sum(1 for w in sampling_weights if w > 2 * uniform_weight)
        undersampled = sum(1 for w in sampling_weights if w < 0.5 * uniform_weight)
        normal = len(sampling_weights) - oversampled - undersampled
        
        f.write(f"\nDialogues oversampled (>2x uniform): {oversampled}/{len(dialogue_counts)}\n")
        f.write(f"Dialogues undersampled (<0.5x uniform): {undersampled}/{len(dialogue_counts)}\n")
        f.write(f"Dialogues normally sampled: {normal}/{len(dialogue_counts)}\n")
        
        f.write("\n5. RARE LABEL ANALYSIS\n")
        f.write("-" * 20 + "\n")
        rare_labels = [(label, stats) for label, stats in label_changes.items() 
                      if stats['original'] < 10 and stats['original'] > 0]
        
        if rare_labels:
            f.write("Labels with original count < 10:\n")
            for label, stats in sorted(rare_labels, key=lambda x: x[1]['original']):
                change_str = f"+{stats['change']}" if stats['change'] > 0 else str(stats['change'])
                f.write(f"  {label}: {stats['original']} -> {stats['final']} ({change_str})\n")
        else:
            f.write("No rare labels (count < 10) found.\n")
        
        f.write("\n6. DIMENSION BREAKDOWN BY FUNCTION\n")
        f.write("-" * 20 + "\n")
        
        # Group labels by dimension
        dimension_labels = defaultdict(list)
        for label in all_labels:
            if ':' in label:
                dimension = label.split(':')[0]
                dimension_labels[dimension].append(label)
        
        for dimension in sorted(dimension_labels.keys()):
            f.write(f"\n{dimension}:\n")
            dim_labels = dimension_labels[dimension]
            # Sort by function name
            dim_labels_sorted = sorted(dim_labels)
            
            for label in dim_labels_sorted:
                if label in label_changes and label_changes[label]['original'] > 0:
                    stats = label_changes[label]
                    change_str = f"+{stats['change']}" if stats['change'] > 0 else str(stats['change'])
                    percent_str = f"{stats['change_percent']:+.1f}%" if stats['change_percent'] != float('inf') else "N/A"
                    
                    function_name = label.split(':', 1)[1]
                    f.write(f"  {function_name:<40} {stats['original']:<6} -> {stats['final']:<6} ({change_str:>6}, {percent_str:>8})\n")
    
    print(f"Analysis complete! Results written to: {output_file}")
    return label_changes, dimension_changes, total_dialogues_before, total_dialogues_after

if __name__ == "__main__":
    # Configuration - adjust these paths according to your setup
    data_path = "./data"  # Path to your data directory
    vocab_path = "./vocab"  # Path to your vocab directory  
    output_file = "sampling_statistics.txt"
    
    # Run the analysis
    label_changes, dimension_changes, total_before, total_after = analyze_sampling_statistics(
        data_path, vocab_path, output_file
    )
    
    # Print summary to console
    print(f"\nSUMMARY:")
    print(f"Total dialogues before sampling: {total_before}")
    print(f"Total dialogues after sampling: {total_after}")
    print(f"Total unique labels: {len(label_changes)}")
    
    increased = sum(1 for stats in label_changes.values() if stats['change'] > 0)
    decreased = sum(1 for stats in label_changes.values() if stats['change'] < 0)
    unchanged = sum(1 for stats in label_changes.values() if stats['change'] == 0)
    
    print(f"Labels increased: {increased}")
    print(f"Labels decreased: {decreased}") 
    print(f"Labels unchanged: {unchanged}")
    
    print(f"\nDIMENSION SUMMARY:")
    for dimension, stats in sorted(dimension_changes.items()):
        change_str = f"+{stats['change']}" if stats['change'] > 0 else str(stats['change'])
        print(f"  {dimension:<25}: {stats['original']:<6} -> {stats['final']:<6} ({change_str})")
    
    print(f"Detailed analysis saved to: {output_file}")