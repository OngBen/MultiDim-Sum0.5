import torch, re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoConfig
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
# REMEMBER 3.12 for annotate_iso.py, 3.5 for train.py and utils.py
# Define ISO 24617-2 communicative functions (Dimension:Function).
dimensions = {
    "Task": ["Inform", "Agreement", "Disagreement", "Correction", "Answer", 
             "Confirm", "Disconfirm", "Question", "Set-Question",
             "Propositional-Question", "Choice-Question", "Check-Question",
             "Offer", "Address-Offer", "Accept-Offer", "Decline-Offer",
             "Promise", "Request", "Address-Request", "Accept-Request",
             "Decline-Request", "Suggest", "Address-Suggest", "Accept-Suggest",
             "Decline-Suggest", "Instruct"],
    "Auto-Feedback": ["AutoPositive", "AutoNegative"],
    "Allo-Feedback": ["AlloPositive", "AlloNegative", "FeedbackElicitation"],
    "Time-Management": ["Stalling", "Pausing"],
    "Turn-Management": ["Turn-Take", "Turn-Grab", "Turn-Accept", 
                        "Turn-Keep", "Turn-Give", "Turn-Release"],
    "Own-Communication-Management": ["Self-Correction", "Self-Error", "Retraction"],
    "Partner-Communication-Management": ["Completion", "Correct-Misspeaking"],
    "Social-Obligations-Management": ["Init-Greeting", "Return-Greeting", 
                                     "Init-Self-Introduction", "Return-Self-Introduction",
                                     "Apology", "Accept-Apology", "Thanking", "Accept-Thanking",
                                     "Init-Goodbye", "Return-Goodbye"],
    "Discourse-Structuring": ["Interaction-Structuring"],
    "Other": ["Other"]
}

# Flatten into a list of labels "Dimension:Function"
iso_labels = []
for dim, funcs in dimensions.items():
    for func in funcs:
        iso_labels.append(f"{dim}:{func}")

# Backchannel, Assess, Inform, Suggest, Stall, 

# Example mapping: Switchboard DA tags (SWDA) to ISO labels (for illustration).
# A real system should use a comprehensive mapping (e.g. Fang et al. 2012).
# SWBD‑DAMSL tag → list of ISO‑24617‑2 labels ("Dimension:Function")
swda_to_iso = {
    "sd":   ["Task:Inform"],  # Non-opinion statements primarily serve to inform without expressing personal views.
    "b":    ["Auto-Feedback:AutoPositive"],  # Backchannels indicate processing/understanding of previous utterances.
    "sv":   ["Task:Inform"],  # Opinion statements, despite subjective content, still serve as information-providing acts.
    "aa":   ["Task:Agreement"],  # Represents agreement with content and/or acceptance of proposals.
    "%-":   ["Turn-Management:Turn-Release"],  # Indicates giving up the turn, directly corresponding to Turn-Release.
    "ba":   ["Social-Obligations-Management:Thanking"],  # Expressions of appreciation align with the thanking function.
    "qy":   ["Task:Propositional-Question", "Turn-Management:Turn-Take"],  # Questions requiring yes/no answers. | Questions often claim the turn.
    "x":    ["Other:Other"],  # Non-verbal utterances do not necessarily have a communicative function.
    "ny":   ["Task:Confirm"],  # Direct affirmative responses serve to confirm propositions.
    "fc":   ["Social-Obligations-Management:Init-Goodbye"],  # Serves the function of initiating the closing phase of conversation.
    "%":    ["Other:Other"],  # Uninterpretable utterances do not necessarily have a communicative function.
    "qw":   ["Task:Set-Question", "Turn-Management:Turn-Take"],  # Questions seeking specific information. | Questions claim the turn.
    "nn":   ["Task:Disconfirm"],  # Direct negative responses serve to disconfirm propositions.
    "bk":   ["Auto-Feedback:AutoPositive"],  # Shows understanding of previous utterance, mapping to positive auto-feedback.
    "h":    ["Task:Inform"],  # Tentative statements that still provide information, though with uncertainty.
    "qy^d": ["Task:Propositional-Question", "Turn-Management:Turn-Take"],  # Yes-No-question phrased declaratively is still a Yes-No-question.
    "fo_o_fw_\"_by_bc":    ["Other:Other"],  # Utterances that cannot be classified into the other dialogue acts.
    "bh":   ["Allo-Feedback:FeedbackElicitation"],  # Question-form backchannels seek feedback from partner on understanding.
    "^q":   ["Task:Inform"],  # Quotations serve to provide information, though in reported form.
    "bf":   ["Auto-Feedback:AutoPositive"],  # Reformulations serve to indicate one's processing of previous statements.
    "na":   ["Task:Confirm"],  # Non-standard affirmative responses still serve to confirm.
    "ny^e": ["Task:Confirm"],  # Non-standard affirmative responses still serve to confirm.
    "ad":   ["Task:Instruct"],  # Directives instruct the addressee to perform actions.
    "^2":   ["Partner-Communication-Management:Completion"],  # Completing partner's utterance is direct equivalent to Completion function.
    "b^m":  ["Auto-Feedback:AutoPositive"],  # Signals one's recognition of the other speaker's speech.
    "qo":   ["Task:Question", "Turn-Management:Turn-Take"],  # Open-ended questions are questions that cannot be mapped to specific question types.
    "qh":   ["Task:Inform"],  # Semantically still functions as a statement.
    "^h":   ["Time-Management:Stalling"],  # Indicates delay before responding, mapping to time management functions.
    "ar":   ["Task:Disagreement"],  # Explicit rejection.
    "ng":   ["Task:Disconfirm"],  # Non-standard negative responses still serve to disconfirm.
    "nn^e":  ["Task:Disconfirm"],  # Non-standard negative responses still serve to disconfirm.
    "br":   ["Auto-Feedback:AutoNegative"],  # Indicates failure to understand on part of the speaker.
    "no":   ["Task:Answer"],  # Generic answers that don't fit other categories still serve to answer.
    "fp":   ["Social-Obligations-Management:Init-Greeting"],  # Serves the function of initiating conversation.
    "qrr":  ["Task:Choice-Question"],  # Gives alternatives for choice, hence mapping to choice-questions.
    "arp_nd":  ["Task:Disagreement"],  # Negative or rejecting responses to questions or proposals still convey disagreement.
    "t3":   ["Task:Inform"],  # Comments about third parties still serve to inform.
    "oo_co_cc":   ["Task:Offer"],  # Commitments to future actions map to offers.
    "t1":   ["Own-Communication-Management:Self-Correction"],  # Speech directed at self rather than addressee serve to self-correct.
    "bd":   ["Social-Obligations-Management:Accept-Apology"],  # Used to minimize importance when accepting apologies.
    "aap_am":  ["Task:Address-Offer"],  # Tentative acceptance or proposals still serve to address offers.
    "^g":   ["Task:Check-Question", "Turn-Management:Turn-Take"],  # Questions added to statements to seek confirmation.
    "qw^d": ["Task:Set-Question", "Turn-Management:Turn-Take"],  # Wh-question phrased declaratively is still a Wh-question.
    "fa":   ["Social-Obligations-Management:Apology"],  # Direct correspondence to apology function.
    "ft":   ["Social-Obligations-Management:Thanking"],  # Direct correspondence to thanking function.
    "+":    ["Discourse-Structuring:Interaction-Structuring"],  # Continuation serves to control the structure and flow of the conversation.
}


def clean_utterance(utt: str) -> str:
    # 1) remove {...}, [...], <...> (including nested)
    utt = re.sub(r'\{[A-Za-z]+\s*([^}]+)\}', r'\1', utt)
    utt = re.sub(r'\{[^}]*\}', '', utt)
    utt = utt.replace(']', '').replace('[', '').replace('/', '')
    utt = re.sub(r'<<[^>]*>>', '', utt)
    utt = re.sub(r'<[^>]*>', '', utt)

    # 2) remove overlap markers '#', turn–fragment dashes, '*' markers
    utt = utt.replace('#', '')
    utt = re.sub(r'\-{2,}', '', utt)
    utt = utt.replace('*', '')

    # 3) remove any leftover '+' and extra spaces if present
    utt = utt.replace('+', '')
    utt = re.sub(r' {2,}', ' ', utt)
    utt = utt.replace('(', '').replace(')', '')

    # 4) collapse whitespace
    utt = re.sub(r'-\s+', '', utt).strip()
    utt = re.sub(r'-\s*$', '', utt)
    return utt


def prepare_dataset_balanced():
    raw = load_dataset('cgpotts/swda', trust_remote_code=True)
    damsl_feature = raw['train'].features['damsl_act_tag']

    # Collect all valid examples
    all_texts = []
    all_labels = []
    label_counts = defaultdict(int)
    
    for ex in raw['train']:
        text = clean_utterance(ex['text'])
        if not text:
            continue

        damsl_idx = ex['damsl_act_tag']
        damsl_tag = damsl_feature.names[damsl_idx]
        iso_funcs = swda_to_iso.get(damsl_tag)
        
        if not iso_funcs:
            continue
            
        all_texts.append(text)
        label_vec = [1.0 if lab in iso_funcs else 0.0 for lab in iso_labels]
        all_labels.append(label_vec)
        
        for label in iso_funcs:
            label_counts[label] += 1

    print(f"Using full dataset: {len(all_texts)} examples")
    print("Label distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {label}: {count} ({count/len(all_texts)*100:.2f}%)")
    
    ds = Dataset.from_dict({"text": all_texts, "labels": all_labels})
    return DatasetDict({"train": ds})

def train_model_improved(train_dataset, model_name='bert-base-uncased'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Calculate ACTUAL labels that appear in our dataset
    labels_array = np.array(train_dataset["train"]["labels"])
    positive_counts = labels_array.sum(axis=0)
    total_examples = len(labels_array)
    
    # Filter to only labels that actually appear
    valid_label_indices = [i for i, count in enumerate(positive_counts) if count > 0]
    valid_iso_labels = [iso_labels[i] for i in valid_label_indices]
    
    print(f"Using {len(valid_iso_labels)} labels that actually appear in dataset")
    print("Label distribution:")
    for i in valid_label_indices:
        count = positive_counts[i]
        if count > 0:  # Double check
            print(f"  {iso_labels[i]}: {count} ({count/len(labels_array)*100:.2f}%)")
    
    positive_ratios = positive_counts / total_examples
    class_weights = torch.tensor(
        np.sqrt(1.0 / np.clip(positive_ratios, 0.0001, 1.0)),
        dtype=torch.float
    )

    print("Class weights (for existing labels):")
    for i in valid_label_indices:
        weight = class_weights[i].item()
        print(f"  {iso_labels[i]}: {weight:.2f}")

    # Use standard model - no custom class needed
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(iso_labels),  # Keep original number for compatibility
        problem_type="multi_label_classification"
    )
    
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length",
            max_length=128
        )
        tokens["labels"] = examples["labels"]
        return tokens
        
    tokenized = train_dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    
    # Use focal loss or other techniques if needed, but start simple
    args = TrainingArguments(
        output_dir="dialog_act_model_improved_3",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=1000,
        weight_decay=0.01,
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        load_best_model_at_end=False,
        metric_for_best_model="eval_micro_f1",
        greater_is_better=True,
        fp16=True if torch.cuda.is_available() else False,
        report_to="none",
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()
        
        micro_f1 = f1_score(labels, predictions, average='micro', zero_division=0)
        macro_f1 = f1_score(labels, predictions, average='macro', zero_division=0)
        
        return {
            'eval_micro_f1': micro_f1,
            'eval_macro_f1': macro_f1,
        }

    # Custom Trainer handles the weighted loss
    class WeightedLossTrainer(Trainer):
        def __init__(self, *args, class_weights=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
            
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            loss_fct = torch.nn.BCEWithLogitsLoss(
                pos_weight=self.class_weights.to(logits.device)
            )
            loss = loss_fct(logits, labels)
            
            return (loss, outputs) if return_outputs else loss

    train_val_split = tokenized["train"].train_test_split(test_size=0.1, seed=42)
    
    trainer = WeightedLossTrainer(  # Use custom trainer
        model=model,
        args=args,
        train_dataset=train_val_split["train"],
        eval_dataset=train_val_split["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=class_weights  # Pass weights here
    )

    trainer.train()
    return model, tokenizer

def predict_labels_improved(model, tokenizer, utterances, confidence_threshold=0.5):
    """Improved prediction with per-dimension confidence thresholds"""
    device = model.device
    inputs = tokenizer(utterances, truncation=True, padding=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
    
    
    predictions = []
    dim_to_indices = {}
    
    for idx, label in enumerate(iso_labels):
        dim = label.split(':', 1)[0]
        dim_to_indices.setdefault(dim, []).append((idx, label))
    
    for prob_vec in probs:
        selected_labels = []
        used_dimensions = set()
        
        # Strategy 1: Per-dimension max with adaptive threshold
        for dim, indices in dim_to_indices.items():
            dim_probs = [(idx, label, prob_vec[idx].item()) for idx, label in indices]
            best_idx, best_label, best_score = max(dim_probs, key=lambda x: x[2])
            
            if best_score > confidence_threshold:
                selected_labels.append(best_label)
                used_dimensions.add(dim)
        
        # Fallback: If still empty, use SINGLE highest-probability label (safer)
        if not selected_labels:
            overall_idx = int(torch.argmax(prob_vec).item())
            selected_labels.append(iso_labels[overall_idx])
        
        predictions.append(selected_labels)
    
    return predictions

def annotate_file(input_path, output_path, model, tokenizer,
                  sep_token="<EOS>", join_with=" "):
    """
    Read dialogues from `input_path`, each line containing one dialogue where
    utterances are separated by `sep_token`. Annotate each utterance, and write
    one output line per dialogue, where each utterance is replaced by its JSON-set
    of ISO tags, joined by `join_with`.
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:

        for line in fin:
            dialogue = line.strip()
            if not dialogue:
                # preserve empty lines
                fout.write("\n")
                continue

            # 1) split the dialogue into utterances
            utterances = [utt.strip() for utt in dialogue.split(sep_token)]
            utterances = utterances[:-1]
            #print(utterances)

            # 2) annotate each utterance
            #    predict_labels expects a list, so we can batch them:
            batch_preds = predict_labels_improved(model, tokenizer, utterances)

            # 3) format each prediction as a JSON-like set
            formatted_preds = []
            for preds in batch_preds:
                if preds:
                    formatted = "{" + ", ".join(f'"{tag}"' for tag in preds) + "}"
                else:
                    formatted = "{}"
                formatted_preds.append(formatted)

            # 4) re-join annotated utterances into one line
            fout.write(join_with.join(formatted_preds) + "\n")

if __name__ == "__main__":
    model_dir = "dialog_act_model"
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"Loaded existing improved model from {model_dir}")
    except:
        print("Training improved model with balanced data...")
        train_dataset = prepare_dataset_balanced()  # Use balanced version
        model, tokenizer = train_model_improved(train_dataset)
        
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Saved improved model to {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Use improved prediction
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "./data/train/in"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "./data/train/da_iso_improved_3"
    
    # You might want to modify annotate_file to use predict_labels_improved
    annotate_file(input_file, output_file, model, tokenizer)