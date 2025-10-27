import re
import numpy as np
from collections import defaultdict

def parse_results_file(content, file_type="multi"):
    """Parse the results file content and extract metrics for each epoch"""
    # Split by epoch sections using the separator pattern
    sections = re.split(r'==================================================', content)[:10]
    
    results = []
    
    for section in sections:
        if not section.strip():
            continue
            
        epoch_data = {}
        
        # Extract epoch number - look for "Epochs: XX" pattern
        epoch_match = re.search(r'Epochs:\s*(\d+)', section)
        if epoch_match:
            epoch_data['epoch'] = int(epoch_match.group(1))
        else:
            # Skip sections without epoch numbers
            continue
        
        # Extract joint training flag
        joint_match = re.search(r'Joint Training:\s*(True|False)', section)
        if joint_match:
            epoch_data['joint'] = joint_match.group(1) == 'True'
        else:
            # For single-dimensional files, check if it's in the filename or assume from content
            if "joint" in file_type.lower():
                epoch_data['joint'] = True
            else:
                epoch_data['joint'] = False
        
        if file_type == "multi":
            # Multi-dimensional format parsing
            # Extract function-level metrics
            func_metrics = re.search(
                r'=== FUNCTION-LEVEL METRICS \(Recommended\) ===\s*'
                r'Micro-Precision:\s*([\d.]+)%\s*'
                r'Micro-Recall:\s*([\d.]+)%\s*'
                r'Micro-F1:\s*([\d.]+)%\s*'
                r'Macro-Precision:\s*([\d.]+)%\s*'
                r'Macro-Recall:\s*([\d.]+)%\s*'
                r'Macro-F1:\s*([\d.]+)%',
                section
            )
            
            if func_metrics:
                epoch_data.update({
                    'micro_precision': float(func_metrics.group(1)),
                    'micro_recall': float(func_metrics.group(2)),
                    'micro_f1': float(func_metrics.group(3)),
                    'macro_precision': float(func_metrics.group(4)),
                    'macro_recall': float(func_metrics.group(5)),
                    'macro_f1': float(func_metrics.group(6))
                })
            
            # Extract other metrics
            other_metrics = re.search(
                r'=== OTHER METRICS ===\s*'
                r'Hamming Loss:\s*([\d.]+)%\s*'
                r'Exact Match:\s*([\d.]+)%\s*'
                r'Average active dimensions per turn:\s*pred=([\d.]+),\s*true=([\d.]+)',
                section
            )
            
            if other_metrics:
                epoch_data.update({
                    'hamming_loss': float(other_metrics.group(1)),
                    'exact_match': float(other_metrics.group(2)),
                    'pred_dims': float(other_metrics.group(3)),
                    'true_dims': float(other_metrics.group(4))
                })
            
            # Extract dimension-level metrics
            dim_metrics = re.search(
                r'=== DIMENSION-LEVEL METRICS ===\s*'
                r'Micro-Precision:\s*([\d.]+)%\s*'
                r'Micro-Recall:\s*([\d.]+)%\s*'
                r'Micro-F1:\s*([\d.]+)%\s*'
                r'Macro-Precision:\s*([\d.]+)%\s*'
                r'Macro-Recall:\s*([\d.]+)%\s*'
                r'Macro-F1:\s*([\d.]+)%',
                section
            )
            
            if dim_metrics:
                epoch_data.update({
                    'dim_micro_precision': float(dim_metrics.group(1)),
                    'dim_micro_recall': float(dim_metrics.group(2)),
                    'dim_micro_f1': float(dim_metrics.group(3)),
                    'dim_macro_precision': float(dim_metrics.group(4)),
                    'dim_macro_recall': float(dim_metrics.group(5)),
                    'dim_macro_f1': float(dim_metrics.group(6))
                })
            
            # NEW: Parse per-dimension accuracy breakdown
            per_dimension_pattern = r'Per-Dimension Accuracy:\s*(.*?)(?=Per-Function Accuracy:|\n\s*$|\Z)'
            per_dimension_match = re.search(per_dimension_pattern, section, re.DOTALL)
            
            if per_dimension_match:
                per_dimension_text = per_dimension_match.group(1)
                dimension_data = parse_per_dimension_accuracy(per_dimension_text)
                epoch_data['per_dimension'] = dimension_data
            
            # NEW: Parse per-function accuracy breakdown  
            per_function_pattern = r'Per-Function Accuracy:\s*(.*?)(?=\n\s*$|\Z)'
            per_function_match = re.search(per_function_pattern, section, re.DOTALL)
            
            if per_function_match:
                per_function_text = per_function_match.group(1)
                function_data = parse_per_function_accuracy(per_function_text)
                epoch_data['per_function'] = function_data
        
        else:  # Single-dimensional format
            # Extract metrics for single-dimensional format
            func_metrics = re.search(
                r'Micro-Precision:\s*([\d.]+)%\s*'
                r'Micro-Recall:\s*([\d.]+)%\s*'
                r'Micro-F1:\s*([\d.]+)%\s*'
                r'Macro-Precision:\s*([\d.]+)%\s*'
                r'Macro-Recall:\s*([\d.]+)%\s*'
                r'Macro-F1:\s*([\d.]+)%',
                section
            )
            
            if func_metrics:
                epoch_data.update({
                    'micro_precision': float(func_metrics.group(1)),
                    'micro_recall': float(func_metrics.group(2)),
                    'micro_f1': float(func_metrics.group(3)),
                    'macro_precision': float(func_metrics.group(4)),
                    'macro_recall': float(func_metrics.group(5)),
                    'macro_f1': float(func_metrics.group(6))
                })
            
            # Extract other metrics for single-dimensional
            other_metrics = re.search(
                r'Hamming Loss:\s*([\d.]+)%\s*'
                r'Exact Match:\s*([\d.]+)%',
                section
            )
            
            if other_metrics:
                epoch_data.update({
                    'hamming_loss': float(other_metrics.group(1)),
                    'exact_match': float(other_metrics.group(2))
                })
            
            # NEW: Parse per-class breakdown for single-dimensional
            per_class_pattern = r'Per-Class Detailed Breakdown:\s*(.*?)(?=\n\s*$|\Z)'
            per_class_match = re.search(per_class_pattern, section, re.DOTALL)
            
            if per_class_match:
                per_class_text = per_class_match.group(1)
                class_data = parse_per_class_breakdown(per_class_text)
                epoch_data['per_class'] = class_data
            
            # Single-dimensional doesn't have dimension-level metrics or dimension counts
            # Set them to None to indicate they're not available
            epoch_data.update({
                'pred_dims': None,
                'true_dims': None,
                'dim_micro_precision': None,
                'dim_micro_recall': None,
                'dim_micro_f1': None,
                'dim_macro_precision': None,
                'dim_macro_recall': None,
                'dim_macro_f1': None
            })
        
        # Only add if we have at least the basic metrics
        if 'micro_f1' in epoch_data:
            results.append(epoch_data)
    
    return results

def parse_per_dimension_accuracy(text):
    """Parse per-dimension accuracy breakdown for multi-dimensional format"""
    dimensions = {}
    
    # Pattern to match dimension lines
    pattern = r'(\w+(?:-\w+)*)\s*\|\s*TP:\s*(\d+)\s*FP:\s*(\d+)\s*FN:\s*(\d+)\s*\|\s*Total:\s*(\d+)\s*\|\s*P:\s*([\d.]+)%\s*R:\s*([\d.]+)%\s*F1:\s*([\d.]+)%'
    
    matches = re.findall(pattern, text)
    
    for match in matches:
        dimension = match[0]
        dimensions[dimension] = {
            'tp': int(match[1]),
            'fp': int(match[2]),
            'fn': int(match[3]),
            'total': int(match[4]),
            'precision': float(match[5]),
            'recall': float(match[6]),
            'f1': float(match[7])
        }
    
    return dimensions

def parse_per_function_accuracy(text):
    """Parse per-function accuracy breakdown for multi-dimensional format"""
    functions = {}
    
    # Pattern to match function lines
    pattern = r'([\w:-]+)\s*\|\s*TP:\s*(\d+)\s*FP:\s*(\d+)\s*FN:\s*(\d+)\s*\|\s*Total:\s*(\d+)\s*\|\s*P:\s*([\d.]+)%\s*R:\s*([\d.]+)%\s*F1:\s*([\d.]+)%'
    
    matches = re.findall(pattern, text)
    
    for match in matches:
        function = match[0]
        functions[function] = {
            'tp': int(match[1]),
            'fp': int(match[2]),
            'fn': int(match[3]),
            'total': int(match[4]),
            'precision': float(match[5]),
            'recall': float(match[6]),
            'f1': float(match[7])
        }
    
    return functions

def parse_per_class_breakdown(text):
    """Parse per-class breakdown for single-dimensional format"""
    classes = {}
    
    # Pattern to match class lines
    pattern = r'(\w+(?:-\w+)*)\s*\|\s*TP:\s*(\d+)\s*FP:\s*(\d+)\s*FN:\s*(\d+)\s*\|\s*Total:\s*(\d+)\s*\|\s*P:\s*([\d.]+)%\s*R:\s*([\d.]+)%\s*F1:\s*([\d.]+)%'
    
    matches = re.findall(pattern, text)
    
    for match in matches:
        class_name = match[0]
        classes[class_name] = {
            'tp': int(match[1]),
            'fp': int(match[2]),
            'fn': int(match[3]),
            'total': int(match[4]),
            'precision': float(match[5]),
            'recall': float(match[6]),
            'f1': float(match[7])
        }
    
    return classes

def calculate_averages(results):
    """Calculate average metrics from all epochs"""
    if not results:
        print("Warning: No results found to calculate averages")
        return {}
    
    avg_metrics = {}
    all_keys = set()
    
    # Collect all metric keys
    for epoch in results:
        all_keys.update(epoch.keys())
    
    # Calculate averages for numeric metrics (skip None values and breakdown dictionaries)
    breakdown_keys = ['per_dimension', 'per_function', 'per_class']  # Keys to exclude from main averaging
    
    for key in all_keys:
        if key in ['epoch', 'joint'] or key in breakdown_keys:
            continue
            
        # Filter out None values for single-dimensional metrics that aren't available
        values = [epoch[key] for epoch in results if key in epoch and epoch[key] is not None]
        if values:
            # Additional check to ensure we're only averaging numeric values
            if all(isinstance(v, (int, float)) for v in values):
                avg_metrics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    # Calculate averages for per-dimension, per-function, and per-class breakdowns
    if any('per_dimension' in epoch for epoch in results):
        avg_metrics['per_dimension'] = calculate_breakdown_averages(
            [epoch.get('per_dimension', {}) for epoch in results]
        )
    
    if any('per_function' in epoch for epoch in results):
        avg_metrics['per_function'] = calculate_breakdown_averages(
            [epoch.get('per_function', {}) for epoch in results]
        )
    
    if any('per_class' in epoch for epoch in results):
        avg_metrics['per_class'] = calculate_breakdown_averages(
            [epoch.get('per_class', {}) for epoch in results]
        )
    
    # Add count and epoch info
    avg_metrics['num_epochs'] = len(results)
    if results:
        epochs = [epoch.get('epoch', 0) for epoch in results]
        avg_metrics['epoch_info'] = {
            'min': min(epochs),
            'max': max(epochs),
            'mean': np.mean(epochs),
            'std': np.std(epochs)
        }
    
    return avg_metrics

def calculate_breakdown_averages(breakdowns_list):
    """Calculate average metrics for per-dimension, per-function, or per-class breakdowns"""
    if not breakdowns_list or not any(breakdowns_list):
        return {}
    
    # Filter out empty breakdowns
    breakdowns_list = [bd for bd in breakdowns_list if bd]
    
    if not breakdowns_list:
        return {}
    
    # Collect all entities (dimensions, functions, or classes) across all epochs
    all_entities = set()
    for breakdown in breakdowns_list:
        if breakdown:
            all_entities.update(breakdown.keys())
    
    entity_avgs = {}
    
    for entity in all_entities:
        # Collect metrics for this entity across all epochs where it appears
        precisions = []
        recalls = []
        f1s = []
        tps = []
        fps = []
        fns = []
        totals = []
        
        for breakdown in breakdowns_list:
            if breakdown and entity in breakdown:
                metrics = breakdown[entity]
                # Ensure all metrics are numeric before appending
                if all(isinstance(metrics.get(k), (int, float)) for k in ['precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'total']):
                    precisions.append(metrics['precision'])
                    recalls.append(metrics['recall'])
                    f1s.append(metrics['f1'])
                    tps.append(metrics['tp'])
                    fps.append(metrics['fp'])
                    fns.append(metrics['fn'])
                    totals.append(metrics['total'])
        
        if precisions:  # Only add if we have data
            entity_avgs[entity] = {
                'precision': {
                    'mean': np.mean(precisions),
                    'std': np.std(precisions),
                    'min': np.min(precisions),
                    'max': np.max(precisions)
                },
                'recall': {
                    'mean': np.mean(recalls),
                    'std': np.std(recalls),
                    'min': np.min(recalls),
                    'max': np.max(recalls)
                },
                'f1': {
                    'mean': np.mean(f1s),
                    'std': np.std(f1s),
                    'min': np.min(f1s),
                    'max': np.max(f1s)
                },
                'tp': {
                    'mean': np.mean(tps),
                    'std': np.std(tps),
                    'min': np.min(tps),
                    'max': np.max(tps)
                },
                'fp': {
                    'mean': np.mean(fps),
                    'std': np.std(fps),
                    'min': np.min(fps),
                    'max': np.max(fps)
                },
                'fn': {
                    'mean': np.mean(fns),
                    'std': np.std(fns),
                    'min': np.min(fns),
                    'max': np.max(fns)
                },
                'total': {
                    'mean': np.mean(totals),
                    'std': np.std(totals),
                    'min': np.min(totals),
                    'max': np.max(totals)
                }
            }
    
    return entity_avgs

def print_comprehensive_comparison(all_results, names):
    """Print comprehensive comparison between all training approaches"""
    print(f"\n{'='*120}")
    print("COMPREHENSIVE COMPARISON: ALL APPROACHES")
    print(f"{'='*120}")
    
    metrics_to_compare = [
        ('micro_precision', 'Func Micro Precision (%)'),
        ('micro_recall', 'Func Micro Recall (%)'),
        ('micro_f1', 'Func Micro F1 (%)'),
        ('macro_precision', 'Func Macro Precision (%)'),
        ('macro_recall', 'Func Macro Recall (%)'),
        ('macro_f1', 'Func Macro F1 (%)'),
        ('hamming_loss', 'Hamming Loss (%)'),
        ('exact_match', 'Exact Match (%)'),
        ('pred_dims', 'Predicted Dimensions/Turn'),
        ('true_dims', 'True Dimensions/Turn'),
        ('dim_micro_precision', 'Dim Micro Precision (%)'),
        ('dim_micro_recall', 'Dim Micro Recall (%)'),
        ('dim_micro_f1', 'Dim Micro F1 (%)'),
        ('dim_macro_precision', 'Dim Macro Precision (%)'),
        ('dim_macro_recall', 'Dim Macro Recall (%)'),
        ('dim_macro_f1', 'Dim Macro F1 (%)')
    ]
    
    # Calculate column widths
    metric_width = max(len(metric_name) for _, metric_name in metrics_to_compare) + 2
    name_width = max(len(name) for name in names) + 2
    
    # Print header
    header = f"{'Metric':<{metric_width}}"
    for name in names:
        header += f" {name:>{name_width}}"
    print(header)
    
    separator = f"{'-'*metric_width}"
    for _ in names:
        separator += f" {'-'*name_width}"
    print(separator)
    
    # Print each metric
    for metric_key, metric_name in metrics_to_compare:
        row = f"{metric_name:<{metric_width}}"
        for result in all_results:
            if metric_key in result:
                mean_val = result[metric_key]['mean']
                std_val = result[metric_key]['std'] 
                row += f" {mean_val:>{name_width-7}.2f}(±{std_val:.2f})"
            else:
                row += f" {'N/A':>{name_width}}"
        print(row)

       

def print_detailed_stats(avg_metrics, title):
    """Print detailed statistics for a training approach"""
    print(f"\n{title}")
    print(f"{'='*60}")
    
    if not avg_metrics:
        print("No data available")
        return
    
    print(f"Number of epochs analyzed: {avg_metrics.get('num_epochs', 'N/A')}")
    if 'epoch_info' in avg_metrics:
        epoch_info = avg_metrics['epoch_info']
        print(f"Epoch statistics: Range {epoch_info['min']}-{epoch_info['max']}, "
              f"Average: {epoch_info['mean']:.1f} ± {epoch_info['std']:.1f}")
    
    # Function-level metrics
    print(f"\n{'FUNCTION-LEVEL METRICS':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    func_metrics = [
        'micro_precision', 'micro_recall', 'micro_f1',
        'macro_precision', 'macro_recall', 'macro_f1'
    ]
    
    for metric in func_metrics:
        if metric in avg_metrics:
            stats = avg_metrics[metric]
            print(f"{metric:<25} {stats['mean']:>9.2f} {stats['std']:>9.2f} {stats['min']:>9.2f} {stats['max']:>9.2f}")
    
    # Dimension-level metrics (only for multi-dimensional)
    dim_metrics = [
        'dim_micro_precision', 'dim_micro_recall', 'dim_micro_f1',
        'dim_macro_precision', 'dim_macro_recall', 'dim_macro_f1'
    ]
    
    # Check if any dimension metrics exist
    has_dim_metrics = any(metric in avg_metrics for metric in dim_metrics)
    
    if has_dim_metrics:
        print(f"\n{'DIMENSION-LEVEL METRICS':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        
        for metric in dim_metrics:
            if metric in avg_metrics:
                stats = avg_metrics[metric]
                print(f"{metric:<25} {stats['mean']:>9.2f} {stats['std']:>9.2f} {stats['min']:>9.2f} {stats['max']:>9.2f}")
    
    # Other important metrics
    print(f"\n{'OTHER METRICS':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    other_metrics = [
        'exact_match', 'hamming_loss', 'pred_dims', 'true_dims'
    ]
    
    for metric in other_metrics:
        if metric in avg_metrics:
            stats = avg_metrics[metric]
            if stats is not None:
                print(f"{metric:<25} {stats['mean']:>9.2f} {stats['std']:>9.2f} {stats['min']:>9.2f} {stats['max']:>9.2f}")

def print_epoch_analysis(results_dict, names_dict):
    """Print detailed analysis of epochs used"""
    print(f"\n{'='*80}")
    print("EPOCH ANALYSIS")
    print(f"{'='*80}")
    
    for name, results in results_dict.items():
        if results:
            epochs = [r['epoch'] for r in results]
            print(f"\n{names_dict[name]}: {len(epochs)} runs")
            print(f"  Range: {min(epochs)} - {max(epochs)}")
            print(f"  Average: {np.mean(epochs):.1f} ± {np.std(epochs):.1f}")
            print(f"  Specific epochs: {sorted(epochs)}")

def print_precision_recall_analysis(avg_metrics_dict, names_dict):
    """Print detailed precision-recall analysis for all approaches"""
    print(f"\n{'='*120}")
    print("PRECISION-RECALL ANALYSIS - ALL APPROACHES")
    print(f"{'='*120}")
    
    # Function-level analysis
    print("\nFUNCTION-LEVEL METRICS:")
    
    # Calculate column widths
    metric_width = 25
    name_width = max(len(name) for name in names_dict.values()) + 2
    
    header = f"{'Metric':<{metric_width}}"
    for name in names_dict.values():
        header += f" {name:>{name_width}}"
    print(header)
    
    separator = f"{'-'*metric_width}"
    for _ in names_dict:
        separator += f" {'-'*name_width}"
    print(separator)
    
    # Micro precision-recall
    metrics_to_show = [
        ('micro_precision', 'Micro Precision'),
        ('micro_recall', 'Micro Recall'),
        ('micro_f1', 'Micro F1'),
        ('macro_precision', 'Macro Precision'),
        ('macro_recall', 'Macro Recall'),
        ('macro_f1', 'Macro F1')
    ]
    
    for metric_key, metric_name in metrics_to_show:
        row = f"{metric_name:<{metric_width}}"
        for avg_metrics in avg_metrics_dict.values():
            if metric_key in avg_metrics:
                mean_val = avg_metrics[metric_key]['mean']
                std_val = avg_metrics[metric_key]['std']
                row += f" {mean_val:>{name_width-7}.2f}(±{std_val:>4.2f})"
            else:
                row += f" {'N/A':>{name_width}}"
        print(row)

        

# NEW FUNCTIONS FOR DETAILED BREAKDOWNS

def print_detailed_breakdowns(avg_metrics, title):
    """Print detailed breakdowns for per-dimension, per-function, or per-class metrics"""
    if not avg_metrics:
        return
    
    # Print per-dimension breakdown for multi-dimensional
    if 'per_dimension' in avg_metrics and avg_metrics['per_dimension']:
        print(f"\n{title} - PER-DIMENSION BREAKDOWN")
        print(f"{'='*90}")
        print(f"{'Dimension':<35} {'Precision':<15} {'Recall':<15} {'F1':<15} {'Support':<10}")
        print(f"{'-'*35} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
        
        dimensions = avg_metrics['per_dimension']
        # Sort by F1 score descending
        sorted_dims = sorted(dimensions.items(), key=lambda x: x[1]['f1']['mean'], reverse=True)
        
        for dim_name, metrics in sorted_dims:
            prec_str = f"{metrics['precision']['mean']:>5.2f}(±{metrics['precision']['std']:>4.2f})"
            rec_str = f"{metrics['recall']['mean']:>5.2f}(±{metrics['recall']['std']:>4.2f})"
            f1_str = f"{metrics['f1']['mean']:>5.2f}(±{metrics['f1']['std']:>4.2f})"
            support_str = f"{metrics['total']['mean']:>9.1f}"
            
            print(f"{dim_name:<35} {prec_str:<15} {rec_str:<15} {f1_str:<15} {support_str:<10}")
    
    # Print per-function breakdown for multi-dimensional
    if 'per_function' in avg_metrics and avg_metrics['per_function']:
        print(f"\n{title} - PER-FUNCTION BREAKDOWN (Top 15 by F1)")
        print(f"{'='*90}")
        print(f"{'Function':<45} {'Precision':<15} {'Recall':<15} {'F1':<15} {'Support':<10}")
        print(f"{'-'*45} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
        
        functions = avg_metrics['per_function']
        # Sort by F1 score descending and take top 15
        sorted_funcs = sorted(functions.items(), key=lambda x: x[1]['f1']['mean'], reverse=True)[:15]
        
        for func_name, metrics in sorted_funcs:
            prec_str = f"{metrics['precision']['mean']:>5.2f}(±{metrics['precision']['std']:>4.2f})"
            rec_str = f"{metrics['recall']['mean']:>5.2f}(±{metrics['recall']['std']:>4.2f})"
            f1_str = f"{metrics['f1']['mean']:>5.2f}(±{metrics['f1']['std']:>4.2f})"
            support_str = f"{metrics['total']['mean']:>9.1f}"
            
            print(f"{func_name:<45} {prec_str:<15} {rec_str:<15} {f1_str:<15} {support_str:<10}")
    
    # Print per-class breakdown for single-dimensional
    if 'per_class' in avg_metrics and avg_metrics['per_class']:
        print(f"\n{title} - PER-CLASS BREAKDOWN")
        print(f"{'='*90}")
        print(f"{'Class':<35} {'Precision':<15} {'Recall':<15} {'F1':<15} {'Support':<10}")
        print(f"{'-'*35} {'-'*15} {'-'*15} {'-'*15} {'-'*10}")
        
        classes = avg_metrics['per_class']
        # Sort by F1 score descending
        sorted_classes = sorted(classes.items(), key=lambda x: x[1]['f1']['mean'], reverse=True)
        
        for class_name, metrics in sorted_classes:
            prec_str = f"{metrics['precision']['mean']:>5.2f}(±{metrics['precision']['std']:>4.2f})"
            rec_str = f"{metrics['recall']['mean']:>5.2f}(±{metrics['recall']['std']:>4.2f})"
            f1_str = f"{metrics['f1']['mean']:>5.2f}(±{metrics['f1']['std']:>4.2f})"
            support_str = f"{metrics['total']['mean']:>9.1f}"
            
            print(f"{class_name:<35} {prec_str:<15} {rec_str:<15} {f1_str:<15} {support_str:<10}")

def print_breakdown_comparison(avg_metrics_dict, names_dict, breakdown_type='per_dimension'):
    """Print comparison of breakdown metrics across approaches"""
    if not any(breakdown_type in metrics for metrics in avg_metrics_dict.values()):
        return
    
    print(f"\n{'='*120}")
    print(f"{breakdown_type.upper().replace('_', ' ')} COMPARISON")
    print(f"{'='*120}")
    
    # Get all entities across all approaches
    all_entities = set()
    for avg_metrics in avg_metrics_dict.values():
        if breakdown_type in avg_metrics:
            all_entities.update(avg_metrics[breakdown_type].keys())
    
    if not all_entities:
        return
    
    # Calculate column widths
    entity_width = max(len(entity) for entity in all_entities) + 2
    name_width = max(len(name) for name in names_dict.values()) + 8  # Extra space for "F1" and formatting
    
    # Print header
    header = f"{'Entity':<{entity_width}}"
    for name in names_dict.values():
        header += f" {name + ' F1':>{name_width}}"
    print(header)
    
    separator = f"{'-'*entity_width}"
    for _ in names_dict:
        separator += f" {'-'*name_width}"
    print(separator)
    
    # Print F1 scores for each entity with mean ± std format
    for entity in (all_entities):
        row = f"{entity:<{entity_width}}"
        for avg_metrics in avg_metrics_dict.values():
            if breakdown_type in avg_metrics and entity in avg_metrics[breakdown_type]:
                f1_mean = avg_metrics[breakdown_type][entity]['f1']['mean']
                f1_std = avg_metrics[breakdown_type][entity]['f1']['std']
                row += f" {f1_mean:>{name_width-7}.2f}(±{f1_std:.2f})"
            else:
                row += f" {'N/A':>{name_width}}"
        print(row)

def main():
    # Read all four files
    file_configs = [
        ('results_multidim_independent.txt', 'multi', 'Multi-Dim Independent'),
        ('results_multidim_joint.txt', 'multi', 'Multi-Dim Joint'),
        ('results_singledim_independent.txt', 'single', 'Single-Dim Independent'),
        ('results_singledim_joint.txt', 'single', 'Single-Dim Joint')
    ]
    
    all_results = {}
    all_avg_metrics = {}
    
    for filename, file_type, display_name in file_configs:
        try:
            with open(filename, 'r') as f:
                content = f.read()
            
            results = parse_results_file(content, file_type)
            all_results[display_name] = results
            all_avg_metrics[display_name] = calculate_averages(results)
            
            print(f"{display_name}: {len(results)} runs parsed")
            
        except FileNotFoundError:
            print(f"Warning: {filename} not found, skipping...")
            all_results[display_name] = []
            all_avg_metrics[display_name] = {}
    
    # Print epoch analysis
    print_epoch_analysis(all_results, {k: k for k in all_results.keys()})
    
    # Print detailed statistics for each approach
    for display_name, avg_metrics in all_avg_metrics.items():
        if avg_metrics:  # Only print if we have data
            print_detailed_stats(avg_metrics, f"{display_name} - DETAILED STATISTICS")
    
    # NEW: Print detailed breakdowns for each approach
    # for display_name, avg_metrics in all_avg_metrics.items():
    #     if avg_metrics:  # Only print if we have data
    #         print_detailed_breakdowns(avg_metrics, display_name)
    
    # Print comprehensive comparison
    valid_metrics = {k: v for k, v in all_avg_metrics.items() if v}  # Only include non-empty results
    if len(valid_metrics) >= 2:
        print_comprehensive_comparison(
            list(valid_metrics.values()), 
            list(valid_metrics.keys())
        )
        
        # Print precision-recall analysis
        print_precision_recall_analysis(valid_metrics, {k: k for k in valid_metrics.keys()})
        
        # NEW: Print breakdown comparisons
        print_breakdown_comparison(valid_metrics, {k: k for k in valid_metrics.keys()}, 'per_dimension')
        print_breakdown_comparison(valid_metrics, {k: k for k in valid_metrics.keys()}, 'per_function')
        print_breakdown_comparison(valid_metrics, {k: k for k in valid_metrics.keys()}, 'per_class')
        
        # Print key insights
        print(f"\n{'='*100}")
        print("KEY INSIGHTS")
        print(f"{'='*100}")

        # Compare multi-dim vs single-dim for joint training
        multi_joint = valid_metrics.get('Multi-Dim Joint')
        single_joint = valid_metrics.get('Single-Dim Joint')
        multi_independent = valid_metrics.get('Multi-Dim Independent')
        single_independent = valid_metrics.get('Single-Dim Independent')

        if multi_joint and single_joint:
            micro_f1_diff = multi_joint['micro_f1']['mean'] - single_joint['micro_f1']['mean']
            multi_std = multi_joint['micro_f1']['std']
            single_std = single_joint['micro_f1']['std']
            print(f"Joint Training - Multi vs Single Dim Micro-F1: {'Multi' if micro_f1_diff > 0 else 'Single'} is better by {abs(micro_f1_diff):.2f}% (Multi: {multi_joint['micro_f1']['mean']:.2f}±{multi_std:.2f}%, Single: {single_joint['micro_f1']['mean']:.2f}±{single_std:.2f}%)")

        if multi_independent and single_independent:
            micro_f1_diff = multi_independent['micro_f1']['mean'] - single_independent['micro_f1']['mean']
            multi_std = multi_independent['micro_f1']['std']
            single_std = single_independent['micro_f1']['std']
            print(f"Independent Training - Multi vs Single Dim Micro-F1: {'Multi' if micro_f1_diff > 0 else 'Single'} is better by {abs(micro_f1_diff):.2f}% (Multi: {multi_independent['micro_f1']['mean']:.2f}±{multi_std:.2f}%, Single: {single_independent['micro_f1']['mean']:.2f}±{single_std:.2f}%)")

        # Compare joint vs independent for each dimension type
        if multi_joint and multi_independent:
            micro_f1_diff = multi_independent['micro_f1']['mean'] - multi_joint['micro_f1']['mean']
            joint_std = multi_joint['micro_f1']['std']
            indep_std = multi_independent['micro_f1']['std']
            print(f"Multi-Dim - Independent vs Joint: {'Independent' if micro_f1_diff > 0 else 'Joint'} is better by {abs(micro_f1_diff):.2f}% (Independent: {multi_independent['micro_f1']['mean']:.2f}±{indep_std:.2f}%, Joint: {multi_joint['micro_f1']['mean']:.2f}±{joint_std:.2f}%)")

        if single_joint and single_independent:
            micro_f1_diff = single_independent['micro_f1']['mean'] - single_joint['micro_f1']['mean']
            joint_std = single_joint['micro_f1']['std']
            indep_std = single_independent['micro_f1']['std']
            print(f"Single-Dim - Independent vs Joint: {'Independent' if micro_f1_diff > 0 else 'Joint'} is better by {abs(micro_f1_diff):.2f}% (Independent: {single_independent['micro_f1']['mean']:.2f}±{indep_std:.2f}%, Joint: {single_joint['micro_f1']['mean']:.2f}±{joint_std:.2f}%)")

if __name__ == "__main__":
    main()