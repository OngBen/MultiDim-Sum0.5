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
                                       "Init-Goodbye", "Return-Goodbye"],
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


def computeMetrics(correct_das, pred_das):
    # Initialize counters
    total_tp_dim = 0  # Dimension-level TP (CORRECTED: only when function is correct)
    total_fp_dim = 0  # Dimension-level FP  
    total_fn_dim = 0  # Dimension-level FN
    
    total_tp_func = 0  # Function-level TP
    total_fp_func = 0  # Function-level FP
    total_fn_func = 0  # Function-level FN
    
    total_turns = 0
    
    # Per-dimension counters (dimension-level)
    dim_tp = [0] * len(dimensions)
    dim_fp = [0] * len(dimensions)
    dim_fn = [0] * len(dimensions)
    dim_total = [0] * len(dimensions)
    
    # Per-function counters (function-level)
    func_tp = {}  # function_string -> count
    func_fp = {}  # function_string -> count  
    func_fn = {}  # function_string -> count
    func_total = {}  # function_string -> count
    
    # Initialize function counters
    for dim in dimensions:
        for func in da_functions[dim]:
            func_key = "{dim}:{func}".format(dim=dim,func=func)
            func_tp[func_key] = 0
            func_fp[func_key] = 0
            func_fn[func_key] = 0
            func_total[func_key] = 0

    # For exact match and hamming loss
    exact_matches = 0
    hamming_errors = 0
    
    # Iterate through all dialogues
    for gold_dialogue, pred_dialogue in zip(correct_das, pred_das):
        # Iterate through all turns in the dialogue
        for turn_idx in range(len(gold_dialogue)):
            total_turns += 1
            gold_turn = gold_dialogue[turn_idx]
            pred_turn = pred_dialogue[turn_idx]
            
            # Check for exact match (both dimension and function must match)
            if gold_turn == pred_turn:
                exact_matches += 1
            
            # Process each dimension
            for dim_idx in range(len(dimensions)):
                gold_label = gold_turn[dim_idx]
                pred_label = pred_turn[dim_idx]
                dim_name = dimensions[dim_idx]
                
                # ===== DIMENSION-LEVEL METRICS =====
                gold_dim_present = (gold_label != 0)
                pred_dim_present = (pred_label != 0)
                
                # Count non-None labels for this dimension
                if gold_dim_present:
                    dim_total[dim_idx] += 1
                
                # CORRECTED: Dimension-level TP/FP/FN
                # A dimension-level TP only occurs when BOTH:
                # 1. The dimension is present in gold AND prediction
                # 2. The specific function is correct
                if gold_dim_present and pred_dim_present:
                    if gold_label == pred_label:  # CORRECT FUNCTION
                        total_tp_dim += 1
                        dim_tp[dim_idx] += 1
                    else:  # WRONG FUNCTION - counts as FP and FN at dimension level
                        total_fp_dim += 1
                        total_fn_dim += 1
                        dim_fp[dim_idx] += 1
                        dim_fn[dim_idx] += 1
                elif not gold_dim_present and pred_dim_present:  # False positive dimension
                    total_fp_dim += 1
                    dim_fp[dim_idx] += 1
                elif gold_dim_present and not pred_dim_present:  # False negative dimension
                    total_fn_dim += 1
                    dim_fn[dim_idx] += 1
                
                # ===== FUNCTION-LEVEL METRICS =====
                if gold_dim_present:
                    gold_func = da_vocab_per_dim[dim_name]['rev'][gold_label]
                    func_key = "{dim_name}:{gold_func}".format(dim_name=dim_name,gold_func=gold_func)
                    func_total[func_key] += 1
                    
                if pred_dim_present:
                    pred_func = da_vocab_per_dim[dim_name]['rev'][pred_label]
                    pred_func_key = "{dim_name}:{pred_func}".format(dim_name=dim_name,pred_func=pred_func)
                
                # Function-level TP/FP/FN
                if gold_dim_present and pred_dim_present:
                    if gold_label == pred_label:  # Same function
                        total_tp_func += 1
                        func_tp[func_key] += 1
                    else:  # Wrong function within dimension
                        total_fp_func += 1
                        total_fn_func += 1
                        func_fp[pred_func_key] += 1
                        func_fn[func_key] += 1
                elif not gold_dim_present and pred_dim_present:  # False positive dimension
                    total_fp_func += 1
                    func_fp[pred_func_key] += 1
                elif gold_dim_present and not pred_dim_present:  # False negative dimension
                    total_fn_func += 1
                    func_fn[func_key] += 1
                
                # Count hamming error (dimension-level)
                if gold_label != pred_label:
                    hamming_errors += 1
    
    # ===== CALCULATE METRICS =====
    
    # Dimension-level micro metrics
    micro_precision_dim = total_tp_dim / (total_tp_dim + total_fp_dim) if (total_tp_dim + total_fp_dim) > 0 else 0
    micro_recall_dim = total_tp_dim / (total_tp_dim + total_fn_dim) if (total_tp_dim + total_fn_dim) > 0 else 0
    micro_f1_dim = 2 * micro_precision_dim * micro_recall_dim / (micro_precision_dim + micro_recall_dim) if (micro_precision_dim + micro_recall_dim) > 0 else 0
    
    # Dimension-level macro metrics
    dim_precisions = []
    dim_recalls = []
    dim_f1s = []
    
    for dim_idx in range(len(dimensions)):
        dim_precision = dim_tp[dim_idx] / (dim_tp[dim_idx] + dim_fp[dim_idx]) if (dim_tp[dim_idx] + dim_fp[dim_idx]) > 0 else 0
        dim_recall = dim_tp[dim_idx] / (dim_tp[dim_idx] + dim_fn[dim_idx]) if (dim_tp[dim_idx] + dim_fn[dim_idx]) > 0 else 0
        dim_f1 = 2 * dim_precision * dim_recall / (dim_precision + dim_recall) if (dim_precision + dim_recall) > 0 else 0
        
        dim_precisions.append(dim_precision)
        dim_recalls.append(dim_recall)
        dim_f1s.append(dim_f1)
    
    macro_precision_dim = sum(dim_precisions) / len(dimensions)
    macro_recall_dim = sum(dim_recalls) / len(dimensions)
    macro_f1_dim = sum(dim_f1s) / len(dimensions)
    
    # Function-level metrics
    micro_precision_func = total_tp_func / (total_tp_func + total_fp_func) if (total_tp_func + total_fp_func) > 0 else 0
    micro_recall_func = total_tp_func / (total_tp_func + total_fn_func) if (total_tp_func + total_fn_func) > 0 else 0
    micro_f1_func = 2 * micro_precision_func * micro_recall_func / (micro_precision_func + micro_recall_func) if (micro_precision_func + micro_recall_func) > 0 else 0
    
    # Function-level macro metrics
    func_precisions = []
    func_recalls = []
    func_f1s = []
    
    for func_key in func_total.keys():
        if func_total[func_key] > 0:
            precision = func_tp[func_key] / (func_tp[func_key] + func_fp[func_key]) if (func_tp[func_key] + func_fp[func_key]) > 0 else 0
            recall = func_tp[func_key] / (func_tp[func_key] + func_fn[func_key]) if (func_tp[func_key] + func_fn[func_key]) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            func_precisions.append(precision)
            func_recalls.append(recall)
            func_f1s.append(f1)
    
    macro_precision_func = sum(func_precisions) / len(func_precisions) if func_precisions else 0
    macro_recall_func = sum(func_recalls) / len(func_recalls) if func_recalls else 0
    macro_f1_func = sum(func_f1s) / len(func_f1s) if func_f1s else 0
    
    # Calculate hamming loss and exact match
    hamming_loss = hamming_errors / (total_turns * len(dimensions))
    exact_match = exact_matches / total_turns if total_turns > 0 else 0
    
    # Calculate per-dimension accuracy and percentages
    dimension_accuracy = []
    dimension_percentages = []
    for dim_idx in range(len(dimensions)):
        if dim_total[dim_idx] > 0:
            accuracy = dim_tp[dim_idx] / dim_total[dim_idx] * 100
        else:
            accuracy = 0.0
        dimension_accuracy.append(accuracy)
        dimension_percentages.append("{:.1f}%".format(accuracy))

    # Create comprehensive formatted function breakdown
    function_counts_formatted = []
    # Sort functions by dimension for better readability
    for dim in dimensions:
        for func in da_functions[dim]:
            func_key = "{dim}:{func}".format(dim=dim,func=func)
            tp = func_tp[func_key]
            fp = func_fp[func_key]
            fn = func_fn[func_key]
            total = func_total[func_key]
            
            if total > 0:
                precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                recall = tp / total * 100
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                breakdown = "{} | TP: {:<3} FP: {:<3} FN: {:<3} | Total: {:<3} | P: {:5.2f}% R: {:5.2f}% F1: {:5.2f}%".format(
                    func_key.ljust(45),  # Pad for alignment
                    tp, fp, fn, total,
                    precision, recall, f1
                )
            else:
                # Only show if there were predictions for this function
                if tp + fp + fn > 0:
                    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                    breakdown = "{} | TP: {:<3} FP: {:<3} FN: {:<3} | Total: {:<3} | P: {:5.2f}% R:   N/A  F1:   N/A".format(
                        func_key.ljust(45),
                        tp, fp, fn, total,
                        precision
                    )
                else:
                    # Skip functions with no occurrences and no predictions
                    continue
                    
            function_counts_formatted.append(breakdown)

    # Create formatted dimension breakdown (similar to single-dimensional)
    dimension_counts_formatted = []
    for dim_idx in range(len(dimensions)):
        dim_name = dimensions[dim_idx]
        tp = dim_tp[dim_idx]
        fp = dim_fp[dim_idx]
        fn = dim_fn[dim_idx]
        total = dim_total[dim_idx]
        
        if total > 0:
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / total * 100
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            breakdown = "{} | TP: {:<4} FP: {:<4} FN: {:<4} | Total: {:<4} | P: {:5.2f}% R: {:5.2f}% F1: {:5.2f}%".format(
                dim_name.ljust(30),
                tp, fp, fn, total,
                precision, recall, f1
            )
        else:
            breakdown = "{} | TP: {:<4} FP: {:<4} FN: {:<4} | Total: {:<4} | P:   N/A  R:   N/A  F1:   N/A".format(
                dim_name.ljust(30),
                tp, fp, fn, total
            )
            
        dimension_counts_formatted.append(breakdown)

    # Return both dimension-level and function-level metrics
    return {
        # Dimension-level metrics (CORRECTED)
        'micro_precision_dim': micro_precision_dim * 100,
        'micro_recall_dim': micro_recall_dim * 100,
        'micro_f1_dim': micro_f1_dim * 100,
        'macro_precision_dim': macro_precision_dim * 100,
        'macro_recall_dim': macro_recall_dim * 100,
        'macro_f1_dim': macro_f1_dim * 100,
        
        # Function-level metrics
        'micro_precision': micro_precision_func * 100,
        'micro_recall': micro_recall_func * 100,
        'micro_f1': micro_f1_func * 100,
        'macro_precision': macro_precision_func * 100,
        'macro_recall': macro_recall_func * 100,
        'macro_f1': macro_f1_func * 100,
        
        'hamming_loss': hamming_loss * 100,
        'exact_match': exact_match * 100,
        'dimension_counts': dim_total,
        'dimension_tp': dim_tp,
        'dimension_fp': dim_fp,
        'dimension_fn': dim_fn,
        'dimension_accuracy': dimension_accuracy,
        'dimension_counts_formatted': dimension_counts_formatted,
        'function_counts_formatted': function_counts_formatted,
        
        # Additional debug info
        'function_level_metrics': {
            'total_tp': total_tp_func,
            'total_fp': total_fp_func,
            'total_fn': total_fn_func
        },
        'dimension_level_metrics': {
            'total_tp': total_tp_dim,
            'total_fp': total_fp_dim,
            'total_fn': total_fn_dim
        }
    }


class DataProcessor(object):
    def __init__(self, in_path, da_path, sum_path, in_vocab, is_training=True):
        self.__fd_in = open(in_path)
        self.__fd_da = open(da_path)
        self.__fd_sum = open(sum_path)
        self.__in_vocab = in_vocab
        self.end = 0
        self.is_training = is_training
        
        # Precompute all data for proper sampling
        if self.is_training:
            self._precompute_all_data()
            self.current_epoch_samples = 0
            self.samples_per_epoch = self._count_total_samples()

    def _count_total_samples(self):
        """Count the total number of samples in the dataset"""
        # Store current position
        current_pos = self.__fd_da.tell()
        self.__fd_da.seek(0)
        
        # Count lines
        count = 0
        while True:
            line = self.__fd_da.readline()
            if not line:
                break
            count += 1
        
        # Restore position
        self.__fd_da.seek(current_pos)
        return count

    def close(self):
        self.current_epoch_samples = 0
        self.end = 0
        self.__fd_in.close(); self.__fd_da.close(); self.__fd_sum.close()

    def _precompute_all_data(self):
        """Precompute all data and class frequencies for proper sampling"""
        print("DEBUG: Starting _precompute_all_data()")
        self.all_data = []
        self.dimension_function_frequencies = {}
        
        # Store current position
        current_pos = self.__fd_da.tell()
        self.__fd_da.seek(0)
        self.__fd_in.seek(0)
        self.__fd_sum.seek(0)

        line_count = 0
        # Read all data and compute dimension-function frequencies
        while True:
            inp = self.__fd_in.readline().strip()
            if not inp:
                break
                
            da_line = self.__fd_da.readline().strip()
            summ = self.__fd_sum.readline().strip()
            
            line_count += 1
            # if line_count % 1000 == 0:
            #     print("DEBUG: Processed {line_count} lines".format(line_count=line_count))

            # Parse all dimension-function pairs in this dialogue
            blocks = re.findall(r'\{([^}]+)\}', da_line)
            dimension_functions = set()
            for block in blocks:
                items = re.split(r'\s*,\s*', block)
                for item in items:
                    if ':' in item:
                        d, f = item.split(':', 1)
                        dimension_functions.add("{}:{}".format(d.strip(), f.strip()))
            
            self.all_data.append({
                'input': inp,
                'da': da_line,
                'summary': summ,
                'dimension_functions': dimension_functions
            })
            
            # Update dimension-function frequencies
            for df in dimension_functions:
                self.dimension_function_frequencies[df] = self.dimension_function_frequencies.get(df, 0) + 1
        
        print("DEBUG: Finished processing {line_count} lines".format(line_count=line_count))

        # Restore position
        self.__fd_da.seek(current_pos)
        self.__fd_in.seek(current_pos)
        self.__fd_sum.seek(current_pos)
        
        # Compute sampling weights based on inverse frequency of dimension-function pairs
        total_utterances = sum(len(re.findall(r'\{([^}]+)\}', data['da'])) for data in self.all_data)
        print("Number of DAs: {}".format(total_utterances))
        total_dialogues = len(self.all_data)
        self.sampling_weights = []
        
        for i, data in enumerate(self.all_data):
            weight = 0.0
            
            # FIX 1: Use smoother inverse frequency
            for df in data['dimension_functions']:
                frequency = self.dimension_function_frequencies[df] / total_utterances
                # Use sqrt(inverse_frequency) for more balanced weighting
                inverse_freq = 1.0 / (frequency + 1e-8)
                weight += np.sqrt(inverse_freq)  # sqrt instead of log
            
            # FIX 2: Better length normalization
            if data['dimension_functions']:
                # Use geometric mean instead of arithmetic mean
                weight = weight / np.sqrt(len(data['dimension_functions']))
            else:
                weight = 1.0
            
            self.sampling_weights.append(weight)
        
        # FIX 3: Better normalization
        # Convert to probabilities while preserving relative differences
        total_weight = sum(self.sampling_weights)
        self.sampling_weights = [w / total_weight for w in self.sampling_weights]
        
        # Debug the actual distribution
        weights_array = np.array(self.sampling_weights)
        uniform_weight = 1.0 / total_dialogues
        rare_count = np.sum(weights_array > 2 * uniform_weight)
        
        print("DEBUG: Effective sampling distribution:")
        print("  Min weight: {:.6f}".format(np.min(weights_array)))
        print("  Max weight: {:.6f}".format(np.max(weights_array))) 
        print("  Uniform weight: {:.6f}".format(uniform_weight))
        print("  Rare dialogues (>2x uniform): {}/{}".format(rare_count, total_dialogues))
        print("  Weight ratio (max/min): {:.1f}x".format(np.max(weights_array) / np.min(weights_array)))

    def get_batch(self, batch_size):
        in_data, da_data, da_weight = [], [], []
        sum_data, length, sum_length = [], [], []
        in_seq, da_seq, sum_seq = [], [], []
        max_len = max_sum = max_word = 0

        batch_in, batch_da, batch_sum, batch_da_wt = [], [], [], []

        # Common processing function for both training and non-training
        def process_dialogue(inp, da_line, summ):
            nonlocal max_len, max_sum, max_word
            
            in_seq.append(inp); da_seq.append(da_line); sum_seq.append(summ)

            inp_ids = sentenceToIds(inp, self.__in_vocab)
            sum_ids = sentenceToIds(summ, self.__in_vocab)

            # parse per-utterance DA blocks
            blocks = re.findall(r'\{([^}]+)\}', da_line)

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
                        if da_map[dim] in da_vocab_per_dim[dim]['vocab']:
                            idx = da_vocab_per_dim[dim]['vocab'][da_map[dim]]
                            ids.append(idx)
                            wts.append(1.0)
                        else:
                            ids.append(0)
                            wts.append(0.0)
                    else:
                        ids.append(0)
                        wts.append(0.1)
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

        # Training path with sampling
        if self.is_training:
            # Sample entire dialogues based on weights
            indices = np.random.choice(
                len(self.all_data), 
                size=batch_size, 
                p=self.sampling_weights,
                replace=True
            )
            
            # Process the selected dialogues
            for idx in indices:
                data = self.all_data[idx]
                process_dialogue(data['input'], data['da'], data['summary'])
                self.current_epoch_samples += 1

            if self.current_epoch_samples >= self.samples_per_epoch:
                self.end = 1
                print("DEBUG: Reached end of epoch")

        # Non-training path (sequential reading)
        else:
            for _ in range(batch_size):
                inp = self.__fd_in.readline().strip()
                if not inp:
                    self.end = 1
                    break
                da_line = self.__fd_da.readline().strip()
                summ = self.__fd_sum.readline().strip()
                process_dialogue(inp, da_line, summ)

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