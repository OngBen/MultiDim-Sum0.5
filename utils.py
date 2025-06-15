# ==================== utils.py ====================
import numpy as np
import os
import re

# ISO 24617-2 dimensions and functions
dimensions = [
    "Task", "Auto-Feedback", "Allo-Feedback", "Time-Management",
    "Turn-Management", "Own-Comm-Management", "Partner-Comm-Management",
    "Discourse-Structuring", "Social-Obligations", "Other"
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
    "Own-Comm-Management": ["Self-Correction", "Self-Error", "Retraction"],
    "Partner-Comm-Management": ["Completion", "Correct-Misspeaking"],
    "Social-Obligations": ["Init-Greeting", "Return-Greeting",
                             "Init-Self-Introduction", "Return-Self-Introduction",
                             "Apology", "Accept-Apology", "Thanking", "Accept-Thanking",
                             "Goodbye", "Return-Goodbye"],
    "Discourse-Structuring": ["Interaction-Structuring"],
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
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    v = vocab['vocab']
    if isinstance(data, str):
        parts = data.split('<EOS>')[:-1]
        words = [p.split() for p in parts]
    else:
        raise TypeError('data must be string with <EOS>')
    ids = []
    for sent in words:
        sent_ids = []
        for w in sent:
            if w.isdigit(): w = '0'
            sent_ids.append(v.get(w, v['_UNK']))
        ids.append(sent_ids)
    return ids


def padSentence(s, max_length, vocab, word_in_sent_length=0):
    if isinstance(s[0], list):
        for _ in range(max_length - len(s)):
            s.append([vocab['vocab']['_PAD']] * word_in_sent_length)
        return s
    else:
        return s + [vocab['vocab']['_PAD']] * (max_length - len(s))


def computeAccuracy(correct_das, pred_das):
    correct = 0
    total = 0
    for gold, pred in zip(correct_das, pred_das):
        for g, p in zip(sorted(gold), sorted(pred)):
            if p == g:
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
        self.__fd_in.close()
        self.__fd_da.close()
        self.__fd_sum.close()

    def get_batch(self, batch_size):
        in_data, da_data, da_weight = [], [], []
        sum_data, sum_weight = [], []
        length, sum_length = [], []
        in_seq, da_seq, sum_seq = [], [], []
        max_len = max_sum = max_word = 0

        batch_in, batch_da, batch_sum = [], [], []
        batch_da_wt = []
        for _ in range(batch_size):
            inp = self.__fd_in.readline()
            da_line = self.__fd_da.readline()
            summ = self.__fd_sum.readline()
            if inp == '' or da_line == '' or summ == '':
                self.end = 1
                break
            inp = inp.strip()
            da_line = da_line.strip()
            summ = summ.strip()
            in_seq.append(inp)
            da_seq.append(da_line)
            sum_seq.append(summ)
            inp_ids = sentenceToIds(inp, self.__in_vocab)
            sum_ids = sentenceToIds(summ, self.__in_vocab)
            pairs = re.findall(r'\{([^}]+)\}', da_line)
            da_map = {dim: None for dim in dimensions}
            for p in pairs:
                if ':' not in p:
                    continue
                d, f = p.split(':', 1)
                if d in da_map:
                    da_map[d] = f
            da_ids, da_wts = [], []
            for dim in dimensions:
                if da_map[dim] is not None:
                    idx = da_vocab_per_dim[dim]['vocab'].get(da_map[dim], 0)
                    da_ids.append(idx)
                    da_wts.append(1.0)
                else:
                    da_ids.append(0)
                    da_wts.append(0.0)
            batch_in.append(np.array(inp_ids))
            batch_sum.append(np.array(sum_ids))
            batch_da.append(np.array(da_ids))
            batch_da_wt.append(np.array(da_wts))
            length.append(len(inp_ids))
            sum_length.append(len(sum_ids))
            max_len = max(max_len, len(inp_ids))
            max_sum = max(max_sum, len(sum_ids))
            max_word = max(max_word, max(len(s) for s in inp_ids))

        for inp_ids, da_ids, da_wts, sum_ids in zip(batch_in, batch_da, batch_da_wt, batch_sum):
            padded_in = [padSentence(list(sent), max_word, self.__in_vocab) for sent in inp_ids]
            in_data.append(padSentence(padded_in, max_len, self.__in_vocab, max_word))
            da_data.append(np.vstack([da_ids] + [[0]*len(dimensions)]*(max_len - 1)))
            da_weight.append(np.vstack([da_wts] + [[0.0]*len(dimensions)]*(max_len - 1)))
            sum_data.append(padSentence(list(sum_ids), max_sum, self.__in_vocab))

        return (
            np.array(in_data), np.array(da_data), np.array(da_weight),
            np.array(length), np.array(sum_data), np.array(da_weight)[:,:,:1],
            np.array(sum_length), in_seq, da_seq, sum_seq
        )

if __name__ == '__main__':
    in_path = './data/valid/in'
    da_path = './data/valid/da_iso'
    sum_path = './data/valid/sum'
    in_vocab = loadVocabulary(os.path.join('./vocab', 'in_vocab'))
    da_vocab = loadVocabulary(os.path.join('./vocab', 'da_vocab'))
    dp = DataProcessor(in_path,da_path,sum_path,in_vocab,da_vocab)
    dp.get_batch(16)