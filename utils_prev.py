import numpy as np
import os
from collections import defaultdict

DA_ISO_MAPPING = {
    'Assess': [('Task', 'Inform'), ('Auto-feedback', 'AutoPositive'), 
               ('Allo-feedback', 'None'), ('Other', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Backchannel': [('Auto-feedback', 'AutoPositive'), ('Turn', 'Accept'), 
                    ('Allo-feedback', 'None'), ('Other', 'None'), ('Task', 'None'), ('Time', 'None')],
    
    'Be-Negative': [('Auto-feedback', 'AutoNegative'),  ('Task', 'Disagreement'),
                    ('Allo-feedback', 'None'), ('Other', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Be-Positive': [('Auto-feedback', 'AutoPositive'),  ('Task', 'Agreement'),
                    ('Allo-feedback', 'None'), ('Other', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Comment-About-Understanding': [('Auto-feedback', 'AutoPositive'), 
                    ('Allo-feedback', 'None'), ('Other', 'None'), ('Task', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Elicit-Assessment': [('Task', 'PropositionalQuestion'), 
                    ('Auto-feedback', 'None'), ('Allo-feedback', 'None'), ('Other', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Elicit-Comment-Understanding': [('Allo-feedback', 'FeedbackElicitation'), 
                    ('Auto-feedback', 'None'), ('Other', 'None'), ('Task', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Elicit-Inform': [('Task', 'SetQuestion'), 
                    ('Auto-feedback', 'None'), ('Allo-feedback', 'None'), ('Other', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Elicit-Offer-Or-Suggestion': [('Task', 'Request'), 
                    ('Auto-feedback', 'None'), ('Allo-feedback', 'None'), ('Other', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Fragment': [('Time', 'Stalling'), 
                    ('Auto-feedback', 'None'), ('Allo-feedback', 'None'), ('Other', 'None'), ('Task', 'None'), ('Turn', 'None')],
    
    'Inform': [('Task', 'Inform'), 
                    ('Auto-feedback', 'None'), ('Allo-feedback', 'None'), ('Other', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Offer': [('Task', 'Offer'), 
                    ('Auto-feedback', 'None'), ('Allo-feedback', 'None'), ('Other', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Other': [('Other', 'Other'), 
                    ('Auto-feedback', 'None'), ('Allo-feedback', 'None'), ('Task', 'None'), ('Time', 'None'), ('Turn', 'None')],
    
    'Stall': [('Time', 'Stalling'), 
                    ('Auto-feedback', 'None'), ('Allo-feedback', 'None'), ('Other', 'None'), ('Task', 'None'), ('Turn', 'None')],
    
    'Suggest': [('Task', 'Suggest'), 
                    ('Auto-feedback', 'None'), ('Allo-feedback', 'None'), ('Other', 'None'), ('Time', 'None'), ('Turn', 'None')]
}

def loadSentenceVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []

    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x,y) for (y,x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}

def loadDAVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    # Collect all unique (dimension, function) pairs
    all_pairs = set()
    for da_type in DA_ISO_MAPPING.values():
        for dim, func in da_type:
            all_pairs.add((dim, func))
    
    # Add special tokens first
    vocab = {'_PAD': 0, '_UNK': 1}
    rev = ['_PAD', '_UNK']
    
    # Sort pairs consistently and assign indices
    sorted_pairs = sorted(all_pairs, key=lambda x: (x[0], x[1]))
    for idx, (dim, func) in enumerate(sorted_pairs, start=2):
        vocab["{dim}:{func}".format(dim=dim, func=func)] = idx
        rev.append("{dim}:{func}".format(dim=dim, func=func))

    # Check if Other:Other is present and add if necessary
    if ('Other:Other' not in vocab) and (('Other', 'Other') not in all_pairs):
        vocab['Other:Other'] = len(vocab)
        rev.append('Other:Other')

    # Build dimension and function mapping
    dimensions = set()
    dim_func_map = defaultdict(list)
    for dim, func in sorted_pairs:
        dimensions.add(dim)
        dim_func_map[dim].append(func)
    # Handle Other:Other if added
    if 'Other:Other' in rev:
        dimensions.add('Other')
        dim_func_map['Other'].append('Other')
    # Sort dimensions and functions
    dimensions = sorted(dimensions)
    for dim in dim_func_map:
        dim_func_map[dim] = sorted(dim_func_map[dim])

    return {
        'vocab': vocab,
        'rev': rev,
        'dimensions': dimensions,
        'dim_func_map': dim_func_map
    }

def sentenceToIds(data, vocab):
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        if data.find('<EOS>') != -1: #input sequence case
            tmp = data.split('<EOS>')[:-1]
            words = []
            for i in tmp:
                words.append(i.split())
        else:
            words = data.split()
    elif isinstance(data, list):
        raise TypeError('list type data is not implement yet')
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for w in words:
        if isinstance(w,list): #input sequence case
            sent = []
            for i in w:
                if str.isdigit(i) == True:
                    i = '0'
                sent.append(vocab.get(i, vocab['_UNK']))
            ids.append(sent)
        else:
            if str.isdigit(w) == True:
                w = '0'
            ids.append(vocab.get(w, vocab['_UNK']))
    return ids

# Modified daToIds
def daToIds(data, vocab):
    if isinstance(data, str):
        if data.find('<EOS>') != -1:
            sentences = data.split('<EOS>')[:-1]
            acts = [sentence.split() for sentence in sentences]
        else:
            acts = [data.split()]
    elif isinstance(data, list):
        acts = data
    else:
        raise TypeError('data should be a string or a list contains words')
    
    ids = []
    for act_list in acts:
        for act in act_list:
            if not act.strip():  # Handle empty tokens
                ids.append([1])  # _UNK
                continue
            # Get DA pairs or default to Other:Other
            pairs = DA_ISO_MAPPING.get(act, [('Other', 'Other')])
            act_indices = []
            for dim, func in pairs:
                key = "{dim}:{func}".format(dim=dim, func=func)
                idx = vocab['vocab'].get(key, 1)  # 1 = _UNK
                act_indices.append(idx)
            if not act_indices:  # Fallback if all invalid
                act_indices.append(1)
            ids.append(act_indices)
    return ids

def daToMultiHot(da_ids, vocab_size):
    batch_size = len(da_ids)
    print(da_ids)
    max_len = max(len(seq) for seq in da_ids)
    multi_hot = np.zeros((batch_size, max_len, vocab_size), dtype=np.float32)
    
    for b, seq in enumerate(da_ids):
        for t, indices in enumerate(seq):
            for idx in indices:
                if idx < vocab_size:  # Avoid index errors
                    multi_hot[b, t, idx] = 1.0
    #print("\nMULTI-HOT ENCODING:\n")
    #print(multi_hot)
    return multi_hot

def padSentence(s, max_length, vocab, word_in_sent_length=0):
    if isinstance(s[0],list): #input sequence case
        for _ in range(max_length-len(s)):
            s.append([vocab['vocab']['_PAD']]*word_in_sent_length)
        return s
    else:
        return s + [vocab['vocab']['_PAD']]*(max_length - len(s))

def padDA(s, max_length, vocab):
    if len(s) > max_length:
        return s[:max_length]
    pad_value = defaultdict(lambda: [vocab['vocab'][dim]['_PAD'] for dim in vocab['vocab']])
    return s + [pad_value] * (max_length - len(s))

# Update computeAccuracy
def computeAccuracy(correct_das, pred_das):
    correct = 0
    total = 0
    for corr_seq, pred_seq in zip(correct_das, pred_das):
        for c, p in zip(corr_seq, pred_seq):
            if c == p:
                correct +=1
            total +=1
    return 100 * correct / total if total >0 else 0

def computeAccuracy2(correct_das, pred_das):
    correctChunkCnt = 0
    foundPredCnt = 0
    
    for correct_da_seq, pred_da_seq in zip(correct_das, pred_das):
        for correct_dict, pred_dict in zip(correct_da_seq, pred_da_seq):
            # Count all key-value pairs in prediction as attempts
            foundPredCnt += len(pred_dict)
            # Check each predicted dimension-function pair
            for pred_dim, pred_func in pred_dict.items():
                # If dimension exists in correct dict and function matches
                if (pred_dim in correct_dict and correct_dict[pred_dim] == pred_func): 
                    correctChunkCnt += 1
    # Calculate precision
    if foundPredCnt > 0:
        precision = 100 * correctChunkCnt / foundPredCnt
    else:
        precision = 0
    #print(precision)
    return precision

def create_da_matrices(da_data, max_dialogue_len):
    """
    Convert dictionary-format dialogue acts into structured numerical arrays.   
    Args:
        da_data: Original dialogue acts with dimension-value pairs
        max_dialogue_len: Maximum length of dialogues for padding
        
    Returns:
        da_data_matrix: Numerical representation of da_data
        dimension_lookup: Dictionary mapping dimensions to indices
    """
    # Get all unique dimensions from DA_ISO_MAPPING
    dimensions = sorted(list(set(dim for _, mappings in DA_ISO_MAPPING.items() 
                               for dim, _ in mappings)))
    dimension_lookup = {dim: idx for idx, dim in enumerate(dimensions)}

    # Initialize matrices with proper shapes
    da_data_matrix = np.zeros((len(da_data), max_dialogue_len, 
                              len(dimensions), 1), dtype=np.int32)
    

    # das_input shape: [batch_size, max_dialogue_len, num_dimensions, num_functions]
    num_functions = 0  # Based on the one-hot vectors in das_input
    
    da_functions = {dim: set() for dim in dimensions}
    for acts in DA_ISO_MAPPING.values():
        for dim, func in acts:
            da_functions[dim].add(func)
            
           
    for dim in da_functions:
        for func in da_functions[dim]:
            num_functions += 1 
    
    # Fill da_data_matrix
    for b, dialogue in enumerate(da_data):
        for t, turn in enumerate(dialogue):           
            if not isinstance(turn, defaultdict):  # Skip padding
                sorted_turn = dict(sorted(turn.items()))
                for dim in sorted_turn:
                    sorted_turn[dim] = sorted(sorted_turn[dim])
                for dim, values in sorted_turn.items():
                    dim_idx = dimension_lookup[dim]
                    #print(dim)
                    #print(sorted_turn)
                    #print(values)
                    for v_idx, value in enumerate(values):
                        da_data_matrix[b, t, dim_idx, v_idx] = value

    #print('da_data:')
    #print(da_data)
    #print('das_input:')
    #print(das_input)  
    #print(da_data_matrix)
    return da_data_matrix, dimension_lookup

class DataProcessor(object):
    def __init__(self, in_path, da_path, sum_path, in_vocab, da_vocab):
        self.__fd_in = open(in_path, 'r')
        self.__fd_da = open(da_path, 'r')
        self.__fd_sum = open(sum_path, 'r')
        self.__in_vocab = in_vocab
        self.__da_vocab = da_vocab
        self.end = 0

    def close(self):
        self.__fd_in.close()
        self.__fd_da.close()
        self.__fd_sum.close()

    def get_batch(self, batch_size):
        in_data = []
        da_data = []
        da_weight = []
        length = []
        sum_data = []
        sum_weight = []
        sum_length = []

        batch_in = []
        batch_da = []
        batch_sum = []
        max_len = 0
        max_sum_len = 0
        max_word_in_sent = 0

        #used to record word(not id)
        in_seq = []
        da_seq = []
        sum_seq = []


        
        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            da = self.__fd_da.readline()
            summ = self.__fd_sum.readline()
            inp = inp.rstrip()
            da = da.rstrip()
            summ = summ.rstrip()

            in_seq.append(inp)
            da_seq.append(da)
            sum_seq.append(summ)

            inp = sentenceToIds(inp, self.__in_vocab)
            da = daToIds(da, self.__da_vocab)
            summ = sentenceToIds(summ, self.__in_vocab)
            batch_in.append(np.array(inp))
            batch_da.append(np.array(da))
            batch_sum.append(np.array(summ))
            length.append(len(inp))
            sum_length.append(len(summ))
            if len(inp) > max_len:
                max_len = len(inp)
            if len(summ) > max_sum_len:
                max_sum_len = len(summ)
            if len(max(inp,key=len)) > max_word_in_sent:
                max_word_in_sent = len(max(inp,key=len))

        #if not batch_in:  # Handle empty batch
        #    return None 
        
        length = np.array(length)
        sum_length = np.array(sum_length)

        for i, s, ints in zip(batch_in, batch_da, batch_sum):
            a = []
            for sent in i:
                a.append(padSentence(list(sent), max_word_in_sent, self.__in_vocab))
            in_data.append(padSentence(list(a), max_len, self.__in_vocab, max_word_in_sent))
            #da_data.append(padSentence(list(s), max_len, self.__da_vocab))
            sum_data.append(padSentence(list(ints), max_sum_len, self.__in_vocab))
        in_data = np.array(in_data)
        #da_data = np.array(da_data)
        #da_weight = np.not_equal(da_data, 0).astype(np.float32)  # 0 = _PAD
        sum_data = np.array(sum_data)

        for i in sum_data:
            weight = np.not_equal(i, np.zeros(i.shape))
            weight = weight.astype(np.float32)
            sum_weight.append(weight)
        sum_weight = np.array(sum_weight)


        da_vocab_size = len(self.__da_vocab['vocab'])
        da_data = daToMultiHot(batch_da, da_vocab_size)
        
        # Pad sequences to max_len
        padded_da = np.zeros((len(batch_da), max_len, da_vocab_size), dtype=np.float32)
        for i, seq in enumerate(da_data):
            padded_da[i, :len(seq)] = seq[:max_len]
        
        da_weight = np.ones((len(batch_da), max_len), dtype=np.float32)  # Mask padding later
        
        

        #da_data, dimension_lookup = create_da_matrices(da_data, max_len)

        #print('in_data:')
        #print(in_data)
        #print('da_data:')
        #print(padded_da)
        #print('da_weight:')
        #print(da_weight)
        #print('length:')
        #print(length)
        #print('sum_data:')
        #print(sum_data)
        #print('sum_weight:')
        #print(sum_weight)
        #print('sum_length:')
        #print(sum_length)
        #print('dimension_lookup')
        #print(dimension_lookup)

        return in_data, padded_da, da_weight, length, sum_data, sum_weight, sum_length, in_seq, da_seq, sum_seq

if __name__ == '__main__':
    in_path = './data/valid/in'
    da_path = './data/valid/da'
    sum_path = './data/valid/sum'
    in_vocab = loadSentenceVocabulary(os.path.join('./vocab', 'in_vocab'))
    da_vocab = loadDAVocabulary(os.path.join('./vocab', 'da_vocab'))
    dp = DataProcessor(in_path,da_path,sum_path,in_vocab,da_vocab)
    dp.get_batch(16)