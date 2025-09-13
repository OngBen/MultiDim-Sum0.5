# ==================== utils.py ====================
import numpy as np
import os
import re

# ISO 24617-2 dimensions and functions (names must match annotation file)
dimensions = [
    "Task",
    "Auto-Feedback",
    "Allo-Feedback",
    "Time-Management",
    "Turn-Management",
    "Own-Communication-Management",
    "Partner-Communication-Management",
    "Discourse-Structuring",
    "Social-Obligations-Management",
    "Other"
]
da_functions = {
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
    "Discourse-Structuring": ["Interaction-Structuring"],
    "Social-Obligations-Management": ["Init-Greeting", "Return-Greeting",
                                       "Init-Self-Introduction", "Return-Self-Introduction",
                                       "Apology", "Accept-Apology", "Thanking", "Accept-Thanking",
                                       "Goodbye", "Return-Goodbye"],
    "Other": ["Other"]
}

# Build per-dimension vocabularies (idx 0 = PAD/None)
da_vocab_per_dim = {}
for dim in dimensions:
    rev = ['_PAD'] + da_functions[dim]
    vocab = {label: idx for idx, label in enumerate(rev)}
    da_vocab_per_dim[dim] = {'vocab': vocab, 'rev': rev}


def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')
    rev = []
    with open(path) as fd:
        for line in fd:
            rev.append(line.strip())
    vocab = {w: i for i, w in enumerate(rev)}
    return {'vocab': vocab, 'rev': rev}


def sentenceToIds(data, vocab):
    v = vocab['vocab']
    if '<EOS>' in data:
        parts = data.split('<EOS>')[:-1]
        words = [p.split() for p in parts]
        ids = []
        for sent in words:
            sent_ids = []
            for w in sent:
                if w.isdigit(): w = '0'
                sent_ids.append(v.get(w, v['_UNK']))
            ids.append(sent_ids)
        return ids
    else:
        tokens = data.split()
        ids = []
        for w in tokens:
            if w.isdigit(): w = '0'
            ids.append(v.get(w, v['_UNK']))
        return ids


def padSentence(s, max_length, vocab, word_in_sent_length=0):
    # Treat empty sequences
    if (isinstance(s, (list, np.ndarray)) and len(s) == 0):
        if isinstance(s, np.ndarray):
            # convert empty array to list
            s = []
        if word_in_sent_length > 0:
            return [[vocab['vocab']['_PAD']] * word_in_sent_length for _ in range(max_length)]
        return [vocab['vocab']['_PAD']] * max_length
    # Pad nested lists (utterance words)
    if isinstance(s, list) and s and isinstance(s[0], list):
        for _ in range(max_length - len(s)):
            s.append([vocab['vocab']['_PAD']] * word_in_sent_length)
        return s
    # Flat list (summary tokens)
    return list(s) + [vocab['vocab']['_PAD']] * (max_length - len(s))


def computeAccuracy(correct_das, pred_das):
    correct = 0
    total = 0
    for gold, pred in zip(correct_das, pred_das):
        # For each turn, compare all dimensions
        for turn_idx in range(len(gold)):
            gold_turn = gold[turn_idx]
            pred_turn = pred[turn_idx]
            # Compare each dimension in the turn
            for dim in range(len(gold_turn)):
                if gold_turn[dim] == pred_turn[dim]:
                    correct += 1
                total += 1
    return 100.0 * correct / total if total else 0.0


class DataProcessor(object):
    def __init__(self, in_path, da_path, sum_path, in_vocab):
        self.__fd_in = open(in_path)
        self.__fd_da = open(da_path)
        self.__fd_sum = open(sum_path)
        self.__in_vocab = in_vocab
        self.end = 0

    def close(self):
        self.__fd_in.close(); self.__fd_da.close(); self.__fd_sum.close()

    def get_batch(self, batch_size):
        in_data, da_data, da_weight = [], [], []
        sum_data, length, sum_length = [], [], []
        in_seq, da_seq, sum_seq = [], [], []
        max_len = max_sum = max_word = 0

        batch_in, batch_da, batch_sum, batch_da_wt = [], [], [], []
        for _ in range(batch_size):
            inp = self.__fd_in.readline().strip()
            if not inp:
                self.end = 1; break
            da_line = self.__fd_da.readline().strip()
            # print(da_line)
            summ = self.__fd_sum.readline().strip()
            in_seq.append(inp); da_seq.append(da_line); sum_seq.append(summ)

            inp_ids = sentenceToIds(inp, self.__in_vocab)
            sum_ids = sentenceToIds(summ, self.__in_vocab)

            # parse per-utterance DA blocks
            blocks = re.findall(r'\{([^}]+)\}', da_line)

            # Add these debug lines:
            # print("RAW da_line:", da_line)
            # print("  # utterances from inp_ids:", len(inp_ids))
            # print("  # DA blocks found:    ", len(blocks))

            da_ids_list = []
            da_wts_list = []
            for block in blocks:
                items = re.split(r'\s*,\s*', block)
                da_map = {dim: None for dim in dimensions}
                for item in items:
                    content = item.strip().strip('"').strip()
                    if ':' not in content: continue
                    d, f = content.split(':', 1)
                    da_map[d] = f
                ids = []
                wts = []
                for dim in dimensions:
                    if da_map[dim] is not None:
                        # Check if the function exists in the vocabulary
                        if da_map[dim] in da_vocab_per_dim[dim]['vocab']:
                            idx = da_vocab_per_dim[dim]['vocab'][da_map[dim]]
                            ids.append(idx)
                            wts.append(1.0)
                        else:
                            # If function doesn't exist, treat as None
                            ids.append(0)
                            wts.append(0.0)
                    else:
                        # Dimension not present, set to None
                        ids.append(0)
                        wts.append(0.0)
                da_ids_list.append(ids)
                da_wts_list.append(wts)

            batch_in.append(np.array(inp_ids))
            batch_da.append(np.array(da_ids_list))
            batch_da_wt.append(np.array(da_wts_list))
            batch_sum.append(np.array(sum_ids))

            length.append(len(inp_ids))
            sum_length.append(len(sum_ids))
            max_len = max(max_len, len(inp_ids))
            max_sum = max(max_sum, len(sum_ids))
            max_word = max(max_word, max(len(s) for s in inp_ids))

        # pad and assemble batch
        for inp_ids, da_ids_arr, da_wts_arr, sum_ids in zip(batch_in, batch_da, batch_da_wt, batch_sum):
            # pad words within each utterance
            padded_in = [padSentence(sent, max_word, self.__in_vocab) for sent in inp_ids]
            # pad turns
            in_data.append(padSentence(padded_in, max_len, self.__in_vocab, max_word))

            # pad da arrays to [max_len, num_dims]
            pad_rows = max_len - da_ids_arr.shape[0]
            if pad_rows > 0:
                da_ids_arr = np.vstack([da_ids_arr, np.zeros((pad_rows, len(dimensions)), dtype=int)])
                da_wts_arr = np.vstack([da_wts_arr, np.zeros((pad_rows, len(dimensions)), dtype=float)])
            da_data.append(da_ids_arr)
            da_weight.append(da_wts_arr)

            # pad summary tokens
            sum_data.append(padSentence(sum_ids, max_sum, self.__in_vocab))

        # build sum_weight mask
        sum_weight = []
        for s in sum_data:
            w = np.not_equal(s, 0).astype(np.float32)
            sum_weight.append(w)

        # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
        # print('in_data:')
        # print(np.array(in_data))
        # print('da_data:')
        # print(np.array(da_data))
        # print('da_weight:')
        # print(np.array(da_weight))
        # print('length:')
        # print(np.array(length))
        # print('sum_data:')
        # print(np.array(sum_data))
        # print('sum_weight:')
        # print(np.array(sum_weight))
        # print('sum_length:')
        # print(np.array(sum_length))
        # np.set_printoptions(threshold=1000, linewidth=75)

        return (
            np.array(in_data), np.array(da_data), np.array(da_weight),
            np.array(length), np.array(sum_data), np.array(sum_weight),
            np.array(sum_length), in_seq, da_seq, sum_seq
        )
    
if __name__ == '__main__':
    in_path = './data/valid/in'
    da_path = './data/valid/da_iso'
    sum_path = './data/valid/sum'
    in_vocab = loadVocabulary(os.path.join('./vocab', 'in_vocab'))
    #da_vocab = loadVocabulary(os.path.join('./vocab', 'da_vocab'))
    dp = DataProcessor(in_path,da_path,sum_path,in_vocab)
    dp.get_batch(16)