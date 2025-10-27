import numpy as np
import os
import re

# Code to extract these statistics from your data:
def analyze_dataset_statistics(data_path):
    total_dialogues = 0
    total_turns = 0
    total_words = 0
    turns_per_dialogue = []
    words_per_turn = []
    
    with open(os.path.join(data_path, 'in'), 'r') as f:
        for line in f:
            dialogue = line.strip()
            if dialogue:
                total_dialogues += 1
                turns = dialogue.split('<EOS>')[:-1]
                turns = [t for t in turns if t.strip()]
                total_turns += len(turns)
                turns_per_dialogue.append(len(turns))
                
                for turn in turns:
                    words = turn.split()
                    words_per_turn.append(len(words))
                    total_words += len(words)
    
    return {
        'total_dialogues': total_dialogues,
        'total_turns': total_turns,
        'total_words': total_words,
        'avg_turns_per_dialogue': np.mean(turns_per_dialogue),
        'std_turns_per_dialogue': np.std(turns_per_dialogue),
        'avg_words_per_turn': np.mean(words_per_turn),
        'std_words_per_turn': np.std(words_per_turn)
    }

# Code to analyze label distribution:
def analyze_label_distribution(da_path, is_multidimensional=True):
    label_counts = {}
    total_labels = 0
    
    with open(da_path, 'r') as f:
        for line in f:
            if is_multidimensional:
                # Parse multidimensional format: {Dim:Func, Dim:Func}
                blocks = re.findall(r'\{([^}]+)\}', line.strip())
                for block in blocks:
                    labels = re.split(r'\s*,\s*', block)
                    for label in labels:
                        if ':' in label:
                            label_counts[label] = label_counts.get(label, 0) + 1
                            total_labels += 1
            else:
                # Parse single-dimensional format
                labels = line.strip().split()
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                    total_labels += 1
    
    # Calculate distribution statistics
    counts = list(label_counts.values())
    return {
        'total_unique_labels': len(label_counts),
        'total_label_instances': total_labels,
        'most_frequent_labels': sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        'least_frequent_labels': sorted(label_counts.items(), key=lambda x: x[1])[:10],
        'imbalance_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf'),
    }

def analyze_label_distribution_multisplit(data_base_path, splits=['train', 'valid', 'test'], da_file_suffix='da_iso', is_multidimensional=True):
    """
    Analyze label distribution across multiple dataset splits
    """
    label_counts = {}
    split_counts = {'train': 0, 'valid': 0, 'test': 0}
    total_labels = 0
    
    for split in splits:
        da_path = os.path.join(data_base_path, split, da_file_suffix)
        
        if not os.path.exists(da_path):
            print(f"Warning: DA file not found at {da_path}")
            continue
            
        with open(da_path, 'r') as f:
            for line in f:
                if is_multidimensional:
                    # Parse multidimensional format: {Dim:Func, Dim:Func}
                    blocks = re.findall(r'\{([^}]+)\}', line.strip())
                    for block in blocks:
                        labels = re.split(r'\s*,\s*', block)
                        for label in labels:
                            if ':' in label:
                                clean_label = label.strip().strip('"')
                                label_counts[clean_label] = label_counts.get(clean_label, 0) + 1
                                split_counts[split] += 1
                                total_labels += 1
                else:
                    # Parse single-dimensional format
                    labels = line.strip().split()
                    for label in labels:
                        clean_label = label.strip()
                        label_counts[clean_label] = label_counts.get(clean_label, 0) + 1
                        split_counts[split] += 1
                        total_labels += 1
    
    # Calculate distribution statistics
    counts = list(label_counts.values())
    
    # Calculate label overlap between splits
    split_label_presence = {}
    for split in splits:
        split_label_presence[split] = set()
        da_path = os.path.join(data_base_path, split, da_file_suffix)
        if os.path.exists(da_path):
            with open(da_path, 'r') as f:
                for line in f:
                    if is_multidimensional:
                        blocks = re.findall(r'\{([^}]+)\}', line.strip())
                        for block in blocks:
                            labels = re.split(r'\s*,\s*', block)
                            for label in labels:
                                if ':' in label:
                                    clean_label = label.strip().strip('"')
                                    split_label_presence[split].add(clean_label)
                    else:
                        labels = line.strip().split()
                        for label in labels:
                            split_label_presence[split].add(label.strip())
    
    # Calculate overlap statistics
    all_labels = set(label_counts.keys())
    train_labels = split_label_presence.get('train', set())
    test_labels = split_label_presence.get('test', set())
    valid_labels = split_label_presence.get('valid', set())
    
    # Labels that appear in all splits
    common_labels = train_labels & test_labels & valid_labels
    
    # Labels that only appear in one split (potential data leakage issue)
    train_only = train_labels - test_labels - valid_labels
    test_only = test_labels - train_labels - valid_labels
    valid_only = valid_labels - train_labels - test_labels
    
    return {
        'total_unique_labels': len(label_counts),
        'total_label_instances': total_labels,
        'split_distribution': split_counts,
        'most_frequent_labels': sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        'least_frequent_labels': sorted(label_counts.items(), key=lambda x: x[1])[:10],
        'imbalance_ratio': max(counts) / min(counts) if min(counts) > 0 else float('inf'),
        'split_label_overlap': {
            'common_to_all_splits': len(common_labels),
            'train_only_labels': len(train_only),
            'test_only_labels': len(test_only),
            'valid_only_labels': len(valid_only),
            'train_label_coverage': len(train_labels) / len(all_labels) * 100,
            'test_label_coverage': len(test_labels) / len(all_labels) * 100,
            'valid_label_coverage': len(valid_labels) / len(all_labels) * 100
        },
        'label_presence_details': {
            'common_labels': list(common_labels),
            'train_only': list(train_only),
            'test_only': list(test_only),
            'valid_only': list(valid_only)
        }
    }

def print_dataset_statistics(stats):
    """Print dataset statistics in a formatted way"""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Total Dialogues:              {stats['total_dialogues']:>10,}")
    print(f"Total Turns:                  {stats['total_turns']:>10,}")
    print(f"Total Words:                  {stats['total_words']:>10,}")
    print("-"*60)
    print(f"Avg Turns per Dialogue:       {stats['avg_turns_per_dialogue']:>10.2f}")
    print(f"Std Turns per Dialogue:       {stats['std_turns_per_dialogue']:>10.2f}")
    print(f"Avg Words per Turn:           {stats['avg_words_per_turn']:>10.2f}")
    print(f"Std Words per Turn:           {stats['std_words_per_turn']:>10.2f}")
    print("="*60 + "\n")

def print_label_distribution(stats):
    """Print label distribution statistics in a formatted way"""
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION STATISTICS")
    print("="*60)
    print(f"Total Unique Labels:          {stats['total_unique_labels']:>10,}")
    print(f"Total Label Instances:        {stats['total_label_instances']:>10,}")
    print(f"Imbalance Ratio:              {stats['imbalance_ratio']:>10.2f}")
    print("-"*60)
    
    print("\nMost Frequent Labels:")
    print(f"{'Label':<40} {'Count':>10}")
    print("-"*60)
    for label, count in stats['most_frequent_labels']:
        print(f"{label:<40} {count:>10,}")
    
    print("\nLeast Frequent Labels:")
    print(f"{'Label':<40} {'Count':>10}")
    print("-"*60)
    for label, count in stats['least_frequent_labels']:
        print(f"{label:<40} {count:>10,}")
    print("="*60 + "\n")

def print_label_distribution_multisplit(multidim_stats):
    print("Overall Label Distribution Analysis:")
    print(f"Total unique labels: {multidim_stats['total_unique_labels']}")
    print(f"Total label instances: {multidim_stats['total_label_instances']}")
    print(f"Split distribution: {multidim_stats['split_distribution']}")
    print(f"Imbalance ratio: {multidim_stats['imbalance_ratio']:.2f}")
    print("\nMost frequent labels:")
    for label, count in multidim_stats['most_frequent_labels']:
        print(f"  {label}: {count}")

    print("\nLeast frequent labels:")
    for label, count in multidim_stats['least_frequent_labels']:
        print(f"  {label}: {count}")
        
# Example usage:
train_stats = analyze_dataset_statistics('./data/train')
print_dataset_statistics(train_stats)

multidim_stats = analyze_label_distribution('./data/train/da_iso_improved_3', is_multidimensional=True)
print_label_distribution(multidim_stats)

test_stats = analyze_dataset_statistics('./data/test')
print_dataset_statistics(test_stats)

multidim_stats = analyze_label_distribution('./data/test/da_iso_improved_3', is_multidimensional=True)
print_label_distribution(multidim_stats)

valid_stats = analyze_dataset_statistics('./data/valid')
print_dataset_statistics(valid_stats)

multidim_stats = analyze_label_distribution('./data/valid/da_iso_improved_3', is_multidimensional=True)
print_label_distribution(multidim_stats)

data_base_path = './data'  # Update this path as needed
multidim_stats = analyze_label_distribution_multisplit(
    data_base_path=data_base_path,
    splits=['train', 'valid', 'test'],
    da_file_suffix='da_iso_improved_3',  # or 'da' for single-dimensional
    is_multidimensional=True
)
print_label_distribution_multisplit(multidim_stats)