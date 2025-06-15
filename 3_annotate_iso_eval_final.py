import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, hamming_loss, cohen_kappa_score
import matplotlib.pyplot as plt

def set_to_category(label_set):
    """Convert a set of labels to a consistent, sorted representation"""
    if not label_set:
        return "EMPTY"
    return "|".join(sorted(label_set))

# Function to parse model label strings into a set of tags
def parse_label_set(label_str):
    """
    Parse a label string formatted like "{Tag1, Tag2}" into a set of tag strings.
    Handles extra whitespace and optional quotes.
    """
    if not isinstance(label_str, str):
        raise ValueError("Label string is not a str or is NaN")
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
        p = p.strip().strip('"').strip("'")
        if p:
            tags.add(p)
    return tags

# Load CSV file
try:
    df = pd.read_csv('chatbot_eval.csv')
except Exception as e:
    print(f"Error loading CSV file: {e}")
    df = pd.DataFrame(columns=['utterance','model_labels','bot_labels','correct?'])

valid_model_annotations = []
valid_ref_annotations = []
chatgpt_correct_list = []
actual_correct_list = []
skipped_rows = 0

for idx, row in df.iterrows():
    model_str = row.get('model_labels', '')
    bot_str = row.get('bot_labels', '')
    correct_flag = row.get('correct?', '')
    # Parse model label set
    try:
        model_annotations = parse_label_set(model_str)
    except Exception:
        # Skip row if model labels cannot be parsed
        skipped_rows += 1
        continue

    # Check bot_labels field presence and format
    if not isinstance(bot_str, str) or not bot_str.strip():
        skipped_rows += 1
        continue
    bot_content = bot_str.strip()
    if bot_content.startswith("{") and bot_content.endswith("}"):
        bot_content = bot_content[1:-1].strip()
    else:
        # Malformed format
        skipped_rows += 1
        continue
    if not bot_content:
        skipped_rows += 1
        continue

    # Determine ChatGPT reference set
    ref_annotations = None
    content_upper = bot_content.upper()
    # Case: CORRECT
    if content_upper == "CORRECT":
        ref_annotations = model_annotations.copy()
    # Case: INCORRECT, followed by corrected tags
    elif content_upper.startswith("INCORRECT"):
        parts = [p.strip() for p in bot_content.split(',')]
        if len(parts) < 2 or parts[0].upper() != "INCORRECT":
            skipped_rows += 1
            continue
        corrected_tags = set()
        for tag in parts[1:]:
            tag_clean = tag.strip().strip('"').strip("'")
            if tag_clean:
                corrected_tags.add(tag_clean)
        if not corrected_tags:
            skipped_rows += 1
            continue
        ref_annotations = corrected_tags
    else:
        # Unrecognized format
        skipped_rows += 1
        continue

    # Parse the 'correct?' column as boolean
    if isinstance(correct_flag, str):
        chatgpt_flag = correct_flag.strip().lower() == 'true'
    else:
        chatgpt_flag = bool(correct_flag)

    # Collect valid sets and flags
    valid_model_annotations.append(model_annotations)
    valid_ref_annotations.append(ref_annotations)
    chatgpt_correct_list.append(chatgpt_flag)
    actual_correct = (model_annotations == ref_annotations)
    actual_correct_list.append(actual_correct)

print(valid_model_annotations)
print(valid_ref_annotations)

# =================================================================
# Cohen's Kappa Calculation (FIXED)
# =================================================================
# Prepare data for agreement calculation


# Check if we have valid data
n_valid = len(valid_model_annotations)
if n_valid == 0:
    print("No valid rows to evaluate.")
else:
    # Compute evaluation metrics
    all_labels = sorted(set().union(*valid_model_annotations, *valid_ref_annotations))
    # Create binary indicator matrices
    y_true = []
    y_pred = []
    for model_annotations, ref_annotations in zip(valid_model_annotations, valid_ref_annotations):
        y_pred.append([1 if label in model_annotations else 0 for label in all_labels])
        y_true.append([1 if label in ref_annotations else 0 for label in all_labels])
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Micro/macro precision, recall, F1
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Jaccard index (average of per-utterance intersection-over-union)
    jaccard_scores = []
    for model_annotations, ref_annotations in zip(valid_model_annotations, valid_ref_annotations):
        if not model_annotations and not ref_annotations:
            jaccard = 1.0
        else:
            intersection = len(model_annotations.intersection(ref_annotations))
            union = len(model_annotations.union(ref_annotations))
            jaccard = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard)
    average_jaccard = sum(jaccard_scores) / len(jaccard_scores)

    # Subset accuracy (exact match of full label set)
    subset_accuracy = accuracy_score(y_true, y_pred)

    # Hamming loss
    ham_loss = hamming_loss(y_true, y_pred)

    # Cohen's kappa between ChatGPT 'correct?' column and actual correctness
    annotator1 = [set_to_category(s) for s in valid_model_annotations]
    annotator2 = [set_to_category(s) for s in valid_ref_annotations]
    #print(annotator1)
    #print(annotator2)
    # Calculate Cohen's Kappa
    cohens_kappa = cohen_kappa_score(annotator1, annotator2)
    #cohens_kappa = cohen_kappa_score(chatgpt_correct_list, actual_correct_list)

    try:
        import krippendorff
        reliability_data = [annotator1, annotator2]
        kripp_alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement='nominal'
        )
    except ImportError:
        kripp_alpha = None
        print("Krippendorff's Alpha skipped: install 'krippendorff' package")
    except Exception as e:
        kripp_alpha = None
        print(f"Error calculating Krippendorff's Alpha: {e}")

    # Print results to console
    print(f"Valid rows evaluated: {n_valid}")
    if skipped_rows:
        print(f"Rows skipped due to formatting issues: {skipped_rows}")
    print("Micro-averaged Precision: {:.4f}".format(precision_micro))
    print("Micro-averaged Recall:    {:.4f}".format(recall_micro))
    print("Micro-averaged F1:        {:.4f}".format(f1_micro))
    print("Macro-averaged Precision: {:.4f}".format(precision_macro))
    print("Macro-averaged Recall:    {:.4f}".format(recall_macro))
    print("Macro-averaged F1:        {:.4f}".format(f1_macro))
    print("Average Jaccard Index:    {:.4f}".format(average_jaccard))
    print("Subset Accuracy:          {:.4f}".format(subset_accuracy))
    print("Hamming Loss:             {:.4f}".format(ham_loss))
    print("Cohen's Kappa (ChatGPT vs actual): {:.4f}".format(cohens_kappa))
    if kripp_alpha is not None:
        print("Krippendorff's Alpha: {:.4f}".format(kripp_alpha))

    # Write summary to text file
    with open('evaluation_report.txt', 'w') as report_file:
        report_file.write(f"Valid rows evaluated: {n_valid}\n")
        if skipped_rows:
            report_file.write(f"Rows skipped due to formatting issues: {skipped_rows}\n")
        report_file.write("Micro-averaged Precision: {:.4f}\n".format(precision_micro))
        report_file.write("Micro-averaged Recall:    {:.4f}\n".format(recall_micro))
        report_file.write("Micro-averaged F1:        {:.4f}\n".format(f1_micro))
        report_file.write("Macro-averaged Precision: {:.4f}\n".format(precision_macro))
        report_file.write("Macro-averaged Recall:    {:.4f}\n".format(recall_macro))
        report_file.write("Macro-averaged F1:        {:.4f}\n".format(f1_macro))
        report_file.write("Average Jaccard Index:    {:.4f}\n".format(average_jaccard))
        report_file.write("Subset Accuracy:          {:.4f}\n".format(subset_accuracy))
        report_file.write("Hamming Loss:             {:.4f}\n".format(ham_loss))
        report_file.write("Cohen's Kappa (ChatGPT vs actual): {:.4f}\n".format(cohens_kappa))
        if kripp_alpha is not None:
            report_file.write("Krippendorff's Alpha: {:.4f}".format(kripp_alpha))

    # Plot bar chart for key metrics
    metrics = {
        'Micro-F1': f1_micro,
        'Macro-F1': f1_macro,
        'Jaccard': average_jaccard,
        'Subset Acc': subset_accuracy
    }
    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(6,4))
    bars = plt.bar(labels, values, color='skyblue')
    plt.ylim(0, 1.0)
    plt.title('Evaluation Metrics')
    plt.ylabel('Score')
    # Annotate bar values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.close()
