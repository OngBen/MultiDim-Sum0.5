from datasets import load_dataset
import re

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
    "br ":   ["Auto-Feedback:AutoNegative"],  # Indicates failure to understand on part of the speaker.
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

print(', '.join(f"{{{{{item}}}}}" for item in sorted(set(item for sublist in swda_to_iso.values() for item in sublist))))

# 1. Download and cache the SwDA dataset
ds = load_dataset("cgpotts/swda")
unknown_tags = set()
# 2. Inspect available columns
print(ds["train"].column_names)
# -> ['act_tag', 'caller', 'conversation_no', 'damsl_act_tag', 'from_caller',
#     'from_caller_birth_year', ..., 'text', 'utterance_index', ...]
damsl_feature = ds['train'].features['damsl_act_tag']
# 3. Iterate through utterances + DAMSL tags

texts, labels = [], []
count = 0
for example in ds["train"]:
    utterance = example["text"]
    utterance = clean_utterance(utterance)
    if not utterance:
        # skip truly empty utterances
        continue
    damsl_idx = example["damsl_act_tag"]
    damsl_tag = damsl_feature.names[damsl_idx] 

    iso_funcs = swda_to_iso.get(damsl_tag)
    if iso_funcs is None:
        unknown_tags.add(damsl_tag)
        print(f"{damsl_tag}\t{utterance}")
        continue   # skip this example
    count+=1
    #print(count)
    label_vector = [1.0 if lab in iso_funcs else 0.0 for lab in iso_labels]
    #print(f"{damsl_tag}\t{utterance}\t{label_vector}")

    texts.append(utterance)

if not texts:
    raise ValueError("No examples after cleaning—check your mapping!")

# after loop, inspect unknowns:
if unknown_tags:
    print("Warning: skipped unmapped DAMSL tags:", sorted(unknown_tags))