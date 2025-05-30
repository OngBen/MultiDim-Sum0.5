import torch, re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
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
                                     "Goodbye", "Return-Goodbye"],
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


def prepare_dataset():
    # load with trust_remote_code so that the Python script you uploaded is used
    raw = load_dataset('cgpotts/swda', trust_remote_code=True)

    # get the ClassLabel feature so we can convert indices to strings
    damsl_feature = raw['train'].features['damsl_act_tag']

    train_texts, train_labels = [], []
    for ex in raw['train']:
        text = ex['text']
        text = clean_utterance(text)
        if not text:
            # skip truly empty utterances
            continue

        damsl_idx = ex['damsl_act_tag']                # integer 0–42
        damsl_tag = damsl_feature.names[damsl_idx]     # string like "sd", "aa", etc.

        iso_funcs = swda_to_iso.get(damsl_tag)
        if not iso_funcs:
            print(f"{damsl_tag}\t{text}")
            # you may choose to skip or assign a default
            continue

        # build a multi-hot vector over your full iso_labels list
        label_vector = [1.0 if lab in iso_funcs else 0.0 for lab in iso_labels]

        train_texts.append(text)
        train_labels.append(label_vector)

    if not train_texts:
        raise ValueError("No examples mapped—check your swbd_damsl_to_iso keys!")

    ds = Dataset.from_dict({"text": train_texts, "labels": train_labels})
    return DatasetDict({"train": ds})

def train_model(train_dataset, model_name='bert-base-uncased'):
    """Fine-tune a transformer model on the training dataset."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(iso_labels), problem_type="multi_label_classification"
    )
    # Tokenize inputs and attach labels
    def tokenize_fn(examples):
        tokens = tokenizer(examples["text"], truncation=True, padding="max_length")
        tokens["labels"] = examples["labels"]
        return tokens
    tokenized = train_dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.remove_columns(["text"])
    # Training arguments (adjust epochs, batch size, etc. as needed)
    args = TrainingArguments(
        output_dir="dialog_act_model",
        num_train_epochs=1,               # Short demo training
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,     # Accumulate gradients if OOM occurs
        fp16=True,                        # Enable mixed-precision training
        logging_steps=200,                # Reduce logging frequency
        learning_rate=5e-5,               # Slightly higher learning rate
        optim="adamw_torch_fused",        # Faster optimizer
        report_to="none"                  # Disable unnecessary logging
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=tokenized["train"],
                      tokenizer=tokenizer)
    trainer.train()
    return model, tokenizer

def predict_labels(model, tokenizer, utterances, threshold=0.5):
    """Predict ISO labels for a list of utterances (multi-label output)."""
    device = model.device  # Get the device the model is on
    inputs = tokenizer(utterances, truncation=True, padding=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Key fix
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
    predictions = []
    # pre‑compute a map: dimension -> list of (label_idx, label_name)
    dim_to_indices = {}
    for idx, lab in enumerate(iso_labels):
        dim = lab.split(':', 1)[0]
        dim_to_indices.setdefault(dim, []).append(idx)

    for prob_vec in probs:
        utter_labels = []
        # for each dimension, pick the best-scoring function
        for dim, indices in dim_to_indices.items():
            # get (idx, prob) pairs for this dimension
            dim_probs = [(i, prob_vec[i].item()) for i in indices]
            best_idx, best_score = max(dim_probs, key=lambda x: x[1])
            if best_score > threshold:
                utter_labels.append(iso_labels[best_idx])

        # Fallback: if no labels passed threshold, pick the single highest-probability label
        if not utter_labels:
            overall_idx = int(torch.argmax(prob_vec).item())
            utter_labels.append(iso_labels[overall_idx])

        predictions.append(utter_labels)
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
            batch_preds = predict_labels(model, tokenizer, utterances)

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
    
    model_dir = "./dialog_act_model"
    
    # Check if saved model exists
    try:
        # Load pre-trained model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print(f"Loaded existing model from {model_dir}")
        
    except:
        # Train and save if no model exists
        print("Training new model...")
        train_dataset = prepare_dataset()
        model, tokenizer = train_model(train_dataset)
        
        # Explicitly save after training
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Saved model/tokenizer to {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)  # Explicitly move model to device
    # Proceed with annotation
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "./data/valid/in"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "./data/valid/da_iso"
    annotate_file(input_file, output_file, model, tokenizer)