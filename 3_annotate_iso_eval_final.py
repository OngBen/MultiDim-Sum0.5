import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss, cohen_kappa_score
import matplotlib.pyplot as plt

# Define valid ISO 24617-2 dimension-function pairs
VALID_TAGS = {
    'Allo-Feedback:FeedbackElicitation', 'Auto-Feedback:AutoNegative', 'Auto-Feedback:AutoPositive',
    'Discourse-Structuring:Interaction-Structuring', 'Other:Other', 'Own-Communication-Management:Self-Correction',
    'Partner-Communication-Management:Completion', 'Social-Obligations-Management:Accept-Apology',
    'Social-Obligations-Management:Apology', 'Social-Obligations-Management:Init-Goodbye',
    'Social-Obligations-Management:Init-Greeting', 'Social-Obligations-Management:Thanking',
    'Task:Address-Offer', 'Task:Agreement', 'Task:Answer', 'Task:Check-Question', 'Task:Choice-Question',
    'Task:Confirm', 'Task:Disagreement', 'Task:Disconfirm', 'Task:Inform', 'Task:Instruct', 'Task:Offer',
    'Task:Propositional-Question', 'Task:Question', 'Task:Set-Question', 'Time-Management:Stalling',
    'Turn-Management:Turn-Release', 'Turn-Management:Turn-Take'
}

def set_to_category(label_set):
    """Convert a set of labels to a consistent, sorted representation"""
    if not label_set:
        return "EMPTY"
    return "|".join(sorted(label_set))

def validate_tags(tag_set):
    """Validate that all tags in the set are valid ISO 24617-2 tags"""
    if not tag_set:
        return set()
    valid_tags = set()
    for tag in tag_set:
        if tag in VALID_TAGS:
            valid_tags.add(tag)
        else:
            print(f"Warning: Invalid tag '{tag}' found and will be ignored")
    return valid_tags

def parse_label_set(label_str):
    """
    Parse a label string formatted like "{Tag1, Tag2}" into a set of tag strings.
    Handles extra whitespace and optional quotes.
    """
    if not isinstance(label_str, str) or pd.isna(label_str):
        return set()
    
    s = label_str.strip()
    if s.startswith("{") and s.endswith("}"):
        s_inner = s[1:-1].strip()
    else:
        s_inner = s
    
    # Remove any wrapping quotes
    s_inner = s_inner.strip()
    if (s_inner.startswith("'") and s_inner.endswith("'")) or (s_inner.startswith('"') and s_inner.endswith('"')):
        s_inner = s_inner[1:-1].strip()
    
    if not s_inner:
        return set()
    
    # Split by comma to get tags
    parts = s_inner.split(',')
    tags = set()
    for p in parts:
        p_clean = p.strip().strip('"').strip("'")
        if p_clean and p_clean not in ['CORRECT', 'INCORRECT']:
            tags.add(p_clean)
    return tags

def parse_bot_evaluation(bot_str):
    """
    Parse the bot evaluation string to extract:
    - correctness judgment (CORRECT/INCORRECT)
    - suggested corrections (if any)
    """
    if not isinstance(bot_str, str) or pd.isna(bot_str):
        return None, set()
    
    s = bot_str.strip()
    if s.startswith("{") and s.endswith("}"):
        s_inner = s[1:-1].strip()
    else:
        s_inner = s
    
    # Check for CORRECT case
    if s_inner.upper() == "CORRECT":
        return "CORRECT", set()
    
    # Check for INCORRECT case with suggestions
    if s_inner.upper().startswith("INCORRECT"):
        parts = [p.strip() for p in s_inner.split(',')]
        if len(parts) < 2:
            return "INCORRECT", set()
        
        corrected_tags = set()
        for tag in parts[1:]:
            tag_clean = tag.strip().strip('"').strip("'")
            if tag_clean and tag_clean not in ['CORRECT', 'INCORRECT']:
                corrected_tags.add(tag_clean)
        return "INCORRECT", corrected_tags
    
    # Handle other cases (like empty or malformed)
    return None, set()

# Load CSV file
try:
    df = pd.read_csv('chatbot_eval_reviewed.csv')
except Exception as e:
    print(f"Error loading CSV file: {e}")
    df = pd.DataFrame(columns=['utterance','model_labels','bot_labels','correct?'])

valid_model_annotations = []
valid_llm_judgments = []  # LLM's correctness judgment
valid_llm_suggestions = []  # LLM's suggested corrections
llm_correctness = []  # LLM's binary judgment (True/False)
skipped_rows = 0
detailed_errors = []

for idx, row in df.iterrows():
    utterance = row.get('utterance', '')
    model_str = row.get('model_labels', '')
    bot_str = row.get('bot_labels', '')
    correct_flag = row.get('correct?', '')
    
    # Parse model labels
    try:
        model_annotations = parse_label_set(model_str)
        model_annotations = validate_tags(model_annotations)
    except Exception as e:
        skipped_rows += 1
        detailed_errors.append(f"Row {idx+1}: Failed to parse model labels: {e}")
        continue
    
    # Parse bot evaluation
    judgment, suggestions = parse_bot_evaluation(bot_str)
    if judgment is None:
        skipped_rows += 1
        detailed_errors.append(f"Row {idx+1}: Failed to parse bot evaluation: '{bot_str}'")
        continue
    
    suggestions = validate_tags(suggestions)
    
    # Parse the 'correct?' column
    if isinstance(correct_flag, str):
        chatgpt_flag = correct_flag.strip().lower() == 'true'
    else:
        chatgpt_flag = bool(correct_flag)
    
    # Collect data
    valid_model_annotations.append(model_annotations)
    valid_llm_judgments.append(judgment)
    valid_llm_suggestions.append(suggestions)
    llm_correctness.append(chatgpt_flag)

# =================================================================
# Evaluation Metrics
# =================================================================
n_valid = len(valid_model_annotations)
if n_valid == 0:
    print("No valid rows to evaluate.")
    if detailed_errors:
        print("Detailed errors:")
        for err in detailed_errors[:10]:  # Show first 10 errors
            print(f"  - {err}")
else:
    print(f"Valid rows evaluated: {n_valid}")
    if skipped_rows:
        print(f"Rows skipped due to formatting issues: {skipped_rows}")
        if detailed_errors:
            print("First 5 errors:")
            for err in detailed_errors[:5]:
                print(f"  - {err}")
    
    # 1. Basic accuracy metrics
    correct_count = sum(1 for judgment in valid_llm_judgments if judgment == "CORRECT")
    accuracy = correct_count / n_valid
    print(f"\n1. LLM Evaluation Results:")
    print(f"   Correct annotations: {correct_count}/{n_valid} ({accuracy:.4f})")
    print(f"   Incorrect annotations: {n_valid - correct_count}/{n_valid} ({1 - accuracy:.4f})")
    
    # 2. Analyze suggestions for incorrect cases
    incorrect_indices = [i for i, judgment in enumerate(valid_llm_judgments) if judgment == "INCORRECT"]
    if incorrect_indices:
        print(f"\n2. Analysis of Incorrect Annotations ({len(incorrect_indices)} cases):")
        
        # Count common correction patterns
        correction_patterns = {}
        for idx in incorrect_indices:
            model_tags = valid_model_annotations[idx]
            suggested_tags = valid_llm_suggestions[idx]
            pattern = f"Model: {sorted(model_tags)} -> LLM: {sorted(suggested_tags)}"
            correction_patterns[pattern] = correction_patterns.get(pattern, 0) + 1
        
        print("   Most common correction patterns:")
        for pattern, count in sorted(correction_patterns.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"     {count}x: {pattern}")
    
    # 3. Per-label analysis
    all_labels = sorted(VALID_TAGS)
    label_accuracy = {}
    label_counts = {}
    
    for label in all_labels:
        correct_with_label = 0
        total_with_label = 0
        
        for i, model_tags in enumerate(valid_model_annotations):
            if label in model_tags:
                total_with_label += 1
                if valid_llm_judgments[i] == "CORRECT":
                    correct_with_label += 1
        
        if total_with_label > 0:
            label_accuracy[label] = correct_with_label / total_with_label
            label_counts[label] = total_with_label
    
    print(f"\n3. Per-Label Accuracy (min 5 occurrences):")
    for label in sorted(label_accuracy.keys(), key=lambda x: label_counts[x], reverse=True):
        if label_counts[label] >= 5:
            print(f"   {label}: {label_accuracy[label]:.4f} ({label_counts[label]} occurrences)")
    
    # 4. Multi-label metrics (treating LLM suggestions as reference for incorrect cases)
    y_true = []
    y_pred = []
    
    for i, model_tags in enumerate(valid_model_annotations):
        if valid_llm_judgments[i] == "CORRECT":
            # For correct cases, reference = model tags (since LLM confirmed they're correct)
            ref_tags = model_tags
        else:
            # For incorrect cases, reference = LLM's suggested tags
            ref_tags = valid_llm_suggestions[i]
        
        y_pred.append([1 if label in model_tags else 0 for label in all_labels])
        y_true.append([1 if label in ref_tags else 0 for label in all_labels])
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Jaccard index
    jaccard_scores = []
    for i, model_tags in enumerate(valid_model_annotations):
        if valid_llm_judgments[i] == "CORRECT":
            ref_tags = model_tags
        else:
            ref_tags = valid_llm_suggestions[i]
        
        if not model_tags and not ref_tags:
            jaccard = 1.0
        else:
            intersection = len(model_tags.intersection(ref_tags))
            union = len(model_tags.union(ref_tags))
            jaccard = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard)
    average_jaccard = sum(jaccard_scores) / len(jaccard_scores)
    
    subset_accuracy = accuracy_score(y_true, y_pred)
    ham_loss = hamming_loss(y_true, y_pred)
    
    print(f"\n4. Multi-label Performance Metrics:")
    print(f"   Micro-averaged Precision: {precision_micro:.4f}")
    print(f"   Micro-averaged Recall:    {recall_micro:.4f}")
    print(f"   Micro-averaged F1:        {f1_micro:.4f}")
    print(f"   Macro-averaged Precision: {precision_macro:.4f}")
    print(f"   Macro-averaged Recall:    {recall_macro:.4f}")
    print(f"   Macro-averaged F1:        {f1_macro:.4f}")
    print(f"   Average Jaccard Index:    {average_jaccard:.4f}")
    print(f"   Subset Accuracy:          {subset_accuracy:.4f}")
    print(f"   Hamming Loss:             {ham_loss:.4f}")
    
    # 5. Agreement Analysis - Cohen's Kappa and Krippendorff's Alpha
    print(f"\n5. Agreement Analysis:")
    
    # Prepare data for agreement metrics
    model_categories = []
    reference_categories = []
    
    for i, model_tags in enumerate(valid_model_annotations):
        if valid_llm_judgments[i] == "CORRECT":
            ref_tags = model_tags
        else:
            ref_tags = valid_llm_suggestions[i]
        
        model_categories.append(set_to_category(model_tags))
        reference_categories.append(set_to_category(ref_tags))
    
    # Cohen's Kappa
    try:
        cohens_kappa = cohen_kappa_score(model_categories, reference_categories)
        print(f"   Cohen's Kappa: {cohens_kappa:.4f}")
        
        # Interpret Kappa value
        if cohens_kappa < 0.20:
            interpretation = "Slight agreement"
        elif cohens_kappa < 0.40:
            interpretation = "Fair agreement"
        elif cohens_kappa < 0.60:
            interpretation = "Moderate agreement"
        elif cohens_kappa < 0.80:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"
        print(f"   Interpretation: {interpretation}")
        
    except Exception as e:
        print(f"   Cohen's Kappa calculation failed: {e}")
        cohens_kappa = None
    
    # Krippendorff's Alpha
    try:
        import krippendorff
        reliability_data = [model_categories, reference_categories]
        kripp_alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement='nominal'
        )
        print(f"   Krippendorff's Alpha: {kripp_alpha:.4f}")
        
        # Interpret Alpha value (using same ranges as Kappa for consistency)
        if kripp_alpha < 0.20:
            interpretation = "Slight agreement"
        elif kripp_alpha < 0.40:
            interpretation = "Fair agreement"
        elif kripp_alpha < 0.60:
            interpretation = "Moderate agreement"
        elif kripp_alpha < 0.80:
            interpretation = "Substantial agreement"
        else:
            interpretation = "Almost perfect agreement"
        print(f"   Interpretation: {interpretation}")
        
    except ImportError:
        kripp_alpha = None
        print("   Krippendorff's Alpha skipped: install 'krippendorff' package")
    except Exception as e:
        kripp_alpha = None
        print(f"   Error calculating Krippendorff's Alpha: {e}")
    
    # 6. Detailed Disagreement Analysis (REPLACES binary agreement)
    print(f"\n6. Detailed Disagreement Analysis:")
    
    # Calculate per-utterance agreement
    exact_matches = 0
    partial_matches = 0
    complete_mismatches = 0
    
    for i in range(n_valid):
        model_tags = valid_model_annotations[i]
        if valid_llm_judgments[i] == "CORRECT":
            ref_tags = model_tags
        else:
            ref_tags = valid_llm_suggestions[i]
        
        if model_tags == ref_tags:
            exact_matches += 1
        elif model_tags & ref_tags:  # Some overlap
            # print(f"{model_tags} vs {ref_tags}")
            partial_matches += 1
        else:
            complete_mismatches += 1
    
    print(f"   Exact matches: {exact_matches}/{n_valid} ({exact_matches/n_valid:.4f})")
    print(f"   Partial matches: {partial_matches}/{n_valid} ({partial_matches/n_valid:.4f})")
    print(f"   Complete mismatches: {complete_mismatches}/{n_valid} ({complete_mismatches/n_valid:.4f})")
    
    # Analyze the most common types of partial matches
    if partial_matches > 0:
        partial_match_patterns = {}
        for i in range(n_valid):
            model_tags = valid_model_annotations[i]
            if valid_llm_judgments[i] == "INCORRECT":
                ref_tags = valid_llm_suggestions[i]
                if model_tags != ref_tags and model_tags & ref_tags:
                    intersection = model_tags & ref_tags
                    model_only = model_tags - ref_tags
                    ref_only = ref_tags - model_tags
                    pattern = f"Shared: {sorted(intersection)} | Model-only: {sorted(model_only)} | LLM-only: {sorted(ref_only)}"
                    partial_match_patterns[pattern] = partial_match_patterns.get(pattern, 0) + 1
        
        if partial_match_patterns:
            print(f"   Most common partial match patterns:")
            for pattern, count in sorted(partial_match_patterns.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"     {count}x: {pattern}")
    
    # Write comprehensive report
    with open('evaluation_report.txt', 'w') as report_file:
        report_file.write("Dialogue Act Annotation Evaluation Report\n")
        report_file.write("=========================================\n\n")
        report_file.write(f"Valid rows evaluated: {n_valid}\n")
        if skipped_rows:
            report_file.write(f"Rows skipped due to formatting issues: {skipped_rows}\n")
        
        report_file.write(f"\nLLM Evaluation Summary:\n")
        report_file.write(f"  Correct annotations: {correct_count}/{n_valid} ({accuracy:.4f})\n")
        report_file.write(f"  Incorrect annotations: {n_valid - correct_count}/{n_valid} ({1 - accuracy:.4f})\n")
        
        report_file.write(f"\nPerformance Metrics:\n")
        report_file.write(f"  Micro-averaged Precision: {precision_micro:.4f}\n")
        report_file.write(f"  Micro-averaged Recall:    {recall_micro:.4f}\n")
        report_file.write(f"  Micro-averaged F1:        {f1_micro:.4f}\n")
        report_file.write(f"  Macro-averaged Precision: {precision_macro:.4f}\n")
        report_file.write(f"  Macro-averaged Recall:    {recall_macro:.4f}\n")
        report_file.write(f"  Macro-averaged F1:        {f1_macro:.4f}\n")
        report_file.write(f"  Average Jaccard Index:    {average_jaccard:.4f}\n")
        report_file.write(f"  Subset Accuracy:          {subset_accuracy:.4f}\n")
        report_file.write(f"  Hamming Loss:             {ham_loss:.4f}\n")
        
        report_file.write(f"\nAgreement Metrics:\n")
        if cohens_kappa is not None:
            report_file.write(f"  Cohen's Kappa: {cohens_kappa:.4f}\n")
        if kripp_alpha is not None:
            report_file.write(f"  Krippendorff's Alpha: {kripp_alpha:.4f}\n")
    
    # Plot key metrics including agreement
    metrics_data = {
        'Accuracy': accuracy,
        'Micro-F1': f1_micro,
        'Macro-F1': f1_macro,
        'Jaccard': average_jaccard
    }
    
    # Add agreement metrics if available
    if cohens_kappa is not None:
        metrics_data["Cohen's Kappa"] = cohens_kappa
    if kripp_alpha is not None:
        metrics_data["Kripp. Alpha"] = kripp_alpha
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics_data.keys(), metrics_data.values(), 
                   color=['lightblue', 'lightgreen', 'lightcoral', 'lightsalmon', 'lightyellow', 'lightcyan'][:len(metrics_data)])
    plt.ylim(0, 1.0)
    plt.title('Model Performance and Agreement Metrics (LLM as Evaluator)')
    plt.ylabel('Score')
    plt.xticks(rotation=45, ha='right')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nReport saved to 'evaluation_report.txt'")
    print(f"Plot saved to 'metrics_plot.png'")