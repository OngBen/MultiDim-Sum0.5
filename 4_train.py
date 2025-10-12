import os
import argparse
import logging
import sys
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell_impl
from utils_now import loadVocabulary, computeMetrics, DataProcessor, da_vocab_per_dim, dimensions
import rouge
import re

parser = argparse.ArgumentParser(allow_abbrev=False)

#Network
parser.add_argument("--num_units", type=int, default=256, help="Network size.", dest='layer_size')
parser.add_argument("--model_type", type=str, default='full', help="""full | summary_only(default)
                                                                    full: full attention model
                                                                    summary_only: summary attention model""")

#Training Environment
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false',dest='early_stop', help="Disable early stop, which is based on dialogue act accuracy and ROUGE-2 of summary.")
parser.add_argument("--patience", type=int, default=5, help="Patience to wait before stop.")
parser.add_argument("--joint_training", action='store_true', default=False, 
                   help="Whether to train jointly with summary generation (default: True). Set to False for independent DA training.")

#Evaluating Model
parser.add_argument("--evaluate", action='store_true',dest='evaluate_model', help="Load checkpoint and evaluate model on valid and test set.")
parser.add_argument("--ckpt", type=str, default='', help="Full path to checkpoint file.")

#Model and Vocab
parser.add_argument("--data_path", type=str, default='./data', help="Path to data folder.")
parser.add_argument("--model_path", type=str, default='./model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")
parser.add_argument("--result_path", type=str, default='./result', help="Path to save result files.")

#Data
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
parser.add_argument("--input_file", type=str, default='in', help="suffix of input file.")
parser.add_argument("--da_file", type=str, default='da_iso_improved_3', help="suffix of dialogue act label file.")
parser.add_argument("--sum_file", type=str, default='sum', help="suffix of summary file.")

arg=parser.parse_args()

#Print arguments
for k,v in sorted(vars(arg).items()):
    print(k,'=',v)
print()

model_type = arg.model_type
if model_type == 'full':
    remove_da_attn = False
elif model_type == 'summary_only':
    remove_da_attn = True
else:
    print('unknown model type!')
    exit(1)

#full path to data will be: ./data + train/test/valid
full_train_path = os.path.join(arg.data_path,arg.train_data_path)
full_test_path = os.path.join(arg.data_path,arg.test_data_path)
full_valid_path = os.path.join(arg.data_path,arg.valid_data_path)

layer_size = arg.layer_size
batch_size = arg.batch_size
print('*'*20+model_type+' '+str(layer_size)+'*'*20)

vocab_path = arg.vocab_path
joint_training = arg.joint_training
in_vocab = loadVocabulary(os.path.join(vocab_path, 'in_vocab'))
da_vocab = loadVocabulary(os.path.join(vocab_path, 'da_vocab'))
NUM_DIMS = len(dimensions)
DA_SIZES = [len(da_vocab_per_dim[d]['vocab']) for d in dimensions]

if joint_training:
    print('JOINT TRAINING WITH SUMMARY GENERATION')
else:
    print('INDEPENDENT TRAINING. DA PREDICTION ONLY.')

# Add this function after the imports and before valid_model
def da_turn_to_string(indices, da_vocab_per_dim):
    """Convert a turn's DA indices to string like {Dimension:Function, ...}"""
    parts = []
    for dim in dimensions:  # Process in sorted order
        idx = indices[dimensions.index(dim)]  # Get the index for this dimension
        if idx != 0:  # Skip PAD
            label = da_vocab_per_dim[dim]['rev'][idx]
            if label not in ['_PAD', 'None', '_UNK']:
                parts.append("{dim}:{label}".format(dim=dim, label=label))
    if parts:
        return "{" + ", ".join(parts) + "}"
    else:
        return "{}"

# Add this function near the top
def sparse_label_smoothed_cross_entropy(labels, logits, weights, smoothing=0.1):
    num_classes = tf.shape(logits)[-1]
    smooth_positives = 1.0 - smoothing
    smooth_negatives = smoothing / tf.cast(num_classes - 1, tf.float32)
    
    one_hot = tf.one_hot(labels, num_classes)
    smoothed_labels = one_hot * smooth_positives + smooth_negatives * (1 - one_hot)
    
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, 
        labels=smoothed_labels
    ) * weights

def createModel(input_data, input_size, sequence_length, da_size, decoder_sequence_length, layer_size = 256, isTraining = True):
    cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)

    if isTraining == True:
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5,
                                             output_keep_prob=0.5)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5,
                                             output_keep_prob=0.5)

    embedding = tf.get_variable('embedding', [input_size, layer_size])
    inputs = tf.nn.embedding_lookup(embedding, input_data)
    inputs = tf.reduce_sum(inputs,2)

    state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, sequence_length=sequence_length, dtype=tf.float32)
    final_state = tf.concat([final_state[0].h, final_state[1].h], 1)
    state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)
    state_shape = state_outputs.get_shape()

    with tf.variable_scope('attention'):
        da_inputs = state_outputs
        if remove_da_attn == False:
            with tf.variable_scope('da_attn'):
                attn_size = state_shape[2].value
                origin_shape = tf.shape(state_outputs)
                hidden = tf.expand_dims(state_outputs, 1)
                hidden_conv = tf.expand_dims(state_outputs, 2)
                #hidden shape = [batch, sentence length, 1, hidden size]
                k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
                hidden_features = tf.nn.conv2d(hidden_conv, k, [1, 1, 1, 1], "SAME")
                hidden_features = tf.reshape(hidden_features, origin_shape)
                hidden_features = tf.expand_dims(hidden_features, 1)
                v = tf.get_variable("AttnV", [attn_size])

                da_inputs_shape = tf.shape(da_inputs)
                da_inputs = tf.reshape(da_inputs, [-1, attn_size])
                y = rnn_cell_impl._linear(da_inputs, attn_size, True)
                y = tf.reshape(y, da_inputs_shape)
                y = tf.expand_dims(y, 2)
                s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [3])
                a = tf.nn.softmax(s)
                #a shape = [batch, input size, sentence length, 1]
                a = tf.expand_dims(a, -1)
                da_d = tf.reduce_sum(a * hidden, [2])
        else:
            attn_size = state_shape[2].value
            da_inputs = tf.reshape(da_inputs, [-1, attn_size])

        sum_input = final_state
        if joint_training:
            with tf.variable_scope('sum_attn'):
                BOS_time_slice = tf.ones([batch_size], dtype=tf.int32, name='BOS') * 2
                BOS_step_embedded = tf.nn.embedding_lookup(embedding, BOS_time_slice)
                pad_step_embedded = tf.zeros([batch_size, layer_size],dtype=tf.float32)

                #helper functions for seq2seq
                def initial_fn():
                    initial_elements_finished = (0 >= decoder_sequence_length)  #all False at the initial step
                    initial_input = BOS_step_embedded
                    return initial_elements_finished, initial_input

                def sample_fn(time, outputs, state):
                    prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
                    return prediction_id

                def next_inputs_fn(time, outputs, state, sample_ids):
                    pred_embedding = tf.nn.embedding_lookup(embedding, sample_ids)
                    next_input = pred_embedding
                    elements_finished = (time >= decoder_sequence_length)  #this operation produces boolean tensor of [batch_size]
                    all_finished = tf.reduce_all(elements_finished)  #-> boolean scalar
                    next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
                    next_state = state
                    return elements_finished, next_inputs, next_state

                my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)

                decoder_cell = tf.contrib.rnn.BasicLSTMCell(final_state.get_shape().as_list()[1])
                if isTraining == True:
                    decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell, input_keep_prob=0.5,
                                                        output_keep_prob=0.5)
                attn_mechanism = tf.contrib.seq2seq.LuongAttention(state_shape[2].value, state_outputs,
                                                        memory_sequence_length=sequence_length)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,attn_mechanism,
                        attention_layer_size=state_shape[2].value,alignment_history=True,name='sum_attention')
                sum_out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, input_size)

                decoder = tf.contrib.seq2seq.BasicDecoder(cell=sum_out_cell, helper=my_helper,
                        initial_state=sum_out_cell.zero_state(dtype=tf.float32, batch_size=batch_size))
                decoder_final_outputs,decoder_final_state,_ = tf.contrib.seq2seq.dynamic_decode(
                        decoder=decoder, impute_finished=True, maximum_iterations=tf.reduce_max(decoder_sequence_length))

                attn = tf.transpose(decoder_final_state.alignment_history.stack(),[1,2,0])
                #sum summary attention vector to [batch, encoder_length]
                attn = tf.reduce_mean(attn,axis=2)
                attn = tf.expand_dims(attn,-1)
                d = tf.reduce_sum(attn*state_outputs,axis=1)
                #add final state to summary
                sum_output = tf.concat([d, sum_input], 1)
        else:
            # For independent training, use state_outputs instead of summary
            # Apply mean pooling to state_outputs to get a fixed-size representation
            d = tf.reduce_mean(state_outputs, axis=1)  # [batch_size, hidden_size]
            # Create a dummy summary output for compatibility
            dummy_output = tf.zeros([tf.shape(input_data)[0], tf.reduce_max(decoder_sequence_length), input_size])
            dummy_sample_id = tf.zeros([tf.shape(input_data)[0], tf.reduce_max(decoder_sequence_length)], dtype=tf.int32)
            decoder_final_outputs = type('obj', (object,), {
                'rnn_output': dummy_output,
                'sample_id': dummy_sample_id
            })        

    with tf.variable_scope('sentence_gated'):
        if joint_training:
            sum_gate = rnn_cell_impl._linear(sum_output, attn_size, True)
        else:
            # Use state_outputs instead of summary output for the gate
            # state_mean = tf.reduce_mean(state_outputs, axis=1)
            # sum_gate = rnn_cell_impl._linear(state_mean, attn_size, True)
            sum_gate = tf.zeros([batch_size, attn_size])

        sum_gate = tf.reshape(sum_gate, [-1, 1, sum_gate.get_shape()[1].value])
        v1 = tf.get_variable("gateV", [attn_size])
        if remove_da_attn == False:
            sentence_gate = v1 * tf.tanh(da_d + sum_gate)
        else:
            sentence_gate = v1 * tf.tanh(state_outputs + sum_gate)
        gate_value = tf.reduce_sum(sentence_gate, [2], name='gate_value')
        sentence_gate = tf.expand_dims(gate_value, -1)

        if remove_da_attn == False:
            sentence_gate = da_d * sentence_gate
        else:
            sentence_gate = state_outputs * sentence_gate
        sentence_gate = tf.reshape(sentence_gate, [-1, attn_size])
        da_output = tf.concat([sentence_gate, da_inputs], 1)

    with tf.variable_scope('da_proj'):
        da_logits = []
        for i, dim_size in enumerate(DA_SIZES):
            # use tf.layers.dense to avoid _linear dtype/scope issues
            da_i = tf.layers.dense(da_output, dim_size, name='proj_%d' % i)
            da_logits.append(da_i)
    outputs = da_logits + [decoder_final_outputs.rnn_output, decoder_final_outputs.sample_id]
    return outputs

def valid_model(in_path, da_path, sum_path,sess, save_predictions=False, prediction_prefix=""):
    #return accuracy for dialogue act, rouge-1,2,3,L for summary
    #some useful items are also calculated
    #da_outputs, correct_das: predicted / ground truth of dialogue act

    rouge_1 = []
    rouge_2 = []
    rouge_3 = []
    rouge_L = []
    da_outputs = []
    correct_das = []

    # Debug counters
    num_active_dims_pred = 0
    num_active_dims_true = 0
    num_turns = 0

    if save_predictions:
        # Open files for saving predictions
        # Inside valid_model function, before the open() call:
        os.makedirs(os.path.dirname(prediction_prefix + "_da_true.txt"), exist_ok=True)
        da_true_file = open("{prediction_prefix}_da_true.txt".format(prediction_prefix=prediction_prefix), "w")
        da_pred_file = open("{prediction_prefix}_da_pred.txt".format(prediction_prefix=prediction_prefix), "w")
        sum_true_file = open("{prediction_prefix}_sum_true.txt".format(prediction_prefix=prediction_prefix), "w")
        sum_pred_file = open("{prediction_prefix}_sum_pred.txt".format(prediction_prefix=prediction_prefix), "w")

    data_processor_valid = DataProcessor(in_path, da_path, sum_path, in_vocab, is_training=False)
    while True:
        #get a batch of data
        in_data, da_data, da_weight, length, sums, sum_weight,sum_lengths, in_seq, da_seq, sum_seq = data_processor_valid.get_batch(batch_size)

        if in_data is not None and len(in_data) > 0:
            feed_dict = {input_data.name: in_data, sequence_length.name: length, sum_length.name: sum_lengths}

        if data_processor_valid.end != 1:
            ret = sess.run(inference_outputs, feed_dict)



            #summary part
            pred_sums = []
            correct_sums = []
            # The summary logits are now at index NUM_DIMS (after all DA dimensions)
            summary_logits = ret[NUM_DIMS]

            for batch in summary_logits:
                tmp = []
                for time_i in batch:
                    tmp.append(np.argmax(time_i))
                pred_sums.append(tmp)

            for i in sums:
                correct_sums.append(i.tolist())
            for pred,corr in zip(pred_sums,correct_sums):
                rouge_score_map = rouge.rouge(pred,corr)
                rouge1 = 100*rouge_score_map['rouge_1/f_score']
                rouge2 = 100*rouge_score_map['rouge_2/f_score']
                rouge3 = 100*rouge_score_map['rouge_3/f_score']
                rougeL = 100*rouge_score_map['rouge_l/f_score']
                rouge_1.append(rouge1)
                rouge_2.append(rouge2)
                rouge_3.append(rouge3)
                rouge_L.append(rougeL)

            # Replace the DA processing section with:
            da_logits_list = ret[:NUM_DIMS]  # Get all dimension outputs
            pred_indices_per_dim = []

            # Get the actual batch size from the data
            actual_batch_size = da_data.shape[0]
            max_turns = da_data.shape[1]

            for d in range(NUM_DIMS):
                # Reshape from [batch_size * max_turns, num_classes] to [batch_size, max_turns, num_classes]
                da_logits_d = da_logits_list[d].reshape((actual_batch_size, max_turns, -1))
                pred_indices_per_dim.append(np.argmax(da_logits_d, axis=-1))

            # Stack to [batch, turns, NUM_DIMS]
            pred_das = np.stack(pred_indices_per_dim, axis=-1)

            for i in range(da_data.shape[0]):  # Batch dimension
                dialogue_pred = []
                dialogue_true = []
                for j in range(length[i]):     # Turn dimension
                    # Only include turns with at least one active DA dimension
                    if np.any(da_weight[i, j] > 0):
                        dialogue_pred.append(pred_das[i, j].tolist())
                        # print("Predicted DAs: " + str(pred_das[i, j]))
                        dialogue_true.append(da_data[i, j].tolist())
                        # print("Actual DAs: " + str(da_data[i, j]))
                        # Debug: count active dimensions
                        num_turns += 1
                        num_active_dims_pred += np.count_nonzero(pred_das[i, j])
                        num_active_dims_true += np.count_nonzero(da_data[i, j])
                da_outputs.append(dialogue_pred)
                correct_das.append(dialogue_true)

            # print("DEBUG: ret lengths:", [r.shape if hasattr(r, 'shape') else r for r in ret])
            # print("DEBUG: sample_id shape:", ret[1].shape)
            # print("DEBUG: pred_das shape:", pred_das.shape)
            # print("DEBUG: length:", length)
            # print("DEBUG: da_data shape:", da_data.shape)

            if save_predictions:
                for i in range(len(in_seq)):
                    # Write true DA and summary
                    da_true_file.write(da_seq[i] + "\n")
                    sum_true_file.write(sum_seq[i] + "\n")

                    # Generate predicted DA string
                    pred_da_list = []
                    for j in range(length[i]):
                        turn_indices = pred_das[i, j]
                        pred_da_list.append(da_turn_to_string(turn_indices, da_vocab_per_dim))
                    pred_da_str = " ".join(pred_da_list)
                    da_pred_file.write(pred_da_str + "\n")

                    # Generate predicted summary string
                    pred_sum_indices = pred_sums[i]
                    pred_sum_words = []
                    for idx in pred_sum_indices:
                        if idx != 0:  # Skip PAD tokens
                            if idx < len(in_vocab['rev']):
                                pred_sum_words.append(in_vocab['rev'][idx])
                            else:
                                pred_sum_words.append('_UNK')
                    pred_sum_str = " ".join(pred_sum_words)
                    sum_pred_file.write(pred_sum_str + "\n")

        if data_processor_valid.end == 1:
            break

    if save_predictions:
        da_true_file.close()
        da_pred_file.close()
        sum_true_file.close()
        sum_pred_file.close()

    # Calculate average active dimensions
    if num_turns > 0:
        avg_active_pred = num_active_dims_pred / num_turns
        avg_active_true = num_active_dims_true / num_turns
        print("Average active dimensions per turn: pred={:.2f}, true={:.2f}".format(avg_active_pred, avg_active_true))
    else:
        print("No turns processed for active dimensions calculation.")

        # Replace the existing precision calculation with:
    metrics = computeMetrics(correct_das, da_outputs)
    precision = metrics['micro_precision']  # Keep this for backward compatibility
    metrics['avg_active_pred'] = avg_active_pred
    metrics['avg_active_true'] = avg_active_true

    # Add logging for all metrics
    logging.info('=== Detailed DA Evaluation Metrics ===')
    logging.info('--- FUNCTION-LEVEL ---')
    logging.info('Micro-Precision: {:.2f}%'.format(metrics['micro_precision']))
    logging.info('Micro-Recall: {:.2f}%'.format(metrics['micro_recall']))
    logging.info('Micro-F1: {:.2f}%'.format(metrics['micro_f1']))
    logging.info('Macro-Precision: {:.2f}%'.format(metrics['macro_precision']))
    logging.info('Macro-Recall: {:.2f}%'.format(metrics['macro_recall']))
    logging.info('Macro-F1: {:.2f}%'.format(metrics['macro_f1']))
    logging.info('--- DIMENSION-LEVEL ---')
    logging.info('Micro-Precision: {:.2f}%'.format(metrics['micro_precision_dim']))
    logging.info('Micro-Recall: {:.2f}%'.format(metrics['micro_recall_dim']))
    logging.info('Micro-F1: {:.2f}%'.format(metrics['micro_f1_dim']))
    logging.info('Macro-Precision: {:.2f}%'.format(metrics['macro_precision_dim']))
    logging.info('Macro-Recall: {:.2f}%'.format(metrics['macro_recall_dim']))
    logging.info('Macro-F1: {:.2f}%'.format(metrics['macro_f1_dim']))
    logging.info('--- OTHER METRICS ---')
    logging.info('Hamming Loss: {:.2f}%'.format(metrics['hamming_loss']))
    logging.info('Exact Subset Match: {:.2f}%'.format(metrics['exact_match']))
    logging.info('Function-Level Total: TP: {}, FP: {}, FN: {}'.format(
        metrics['function_level_metrics']['total_tp'],
        metrics['function_level_metrics']['total_fp'],
        metrics['function_level_metrics']['total_fn']
    ))
    logging.info('Per-Dimension Breakdown:')
    for dim_result in metrics['dimension_counts_formatted']:
        logging.info('  {}'.format(dim_result))
    logging.info('Per-Function Detailed Breakdown (showing functions with activity):')
    for func_result in metrics['function_counts_formatted']:
        logging.info('  {}'.format(func_result))
    logging.info('=====================================')
    logging.info('sum rouge1: ' + str(np.mean(rouge_1)))
    logging.info('sum rouge2: ' + str(np.mean(rouge_2)))
    logging.info('sum rouge3: ' + str(np.mean(rouge_3)))
    logging.info('sum rougeL: ' + str(np.mean(rouge_L)))

    data_processor_valid.close()
    return np.mean(rouge_1),np.mean(rouge_2),np.mean(rouge_3),np.mean(rouge_L),metrics

# Placeholders for DA (now [batch, turns, dims])
input_data = tf.placeholder(tf.int32, [None, None, None], name='inputs')
sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
global_step = tf.Variable(0, trainable=False, name='global_step')
# multidim DAS / weights
das = tf.placeholder(tf.int32, [None, None, NUM_DIMS], name='das')
da_weights = tf.placeholder(tf.float32, [None, None, NUM_DIMS], name='da_weights')
# summaries as before
summ = tf.placeholder(tf.int32, [None, None], name='summ')
sum_weights = tf.placeholder(tf.float32, [None, None], name='sum_weights')
sum_length = tf.placeholder(tf.int32, [None], name='sum_length')

# Build training and inference graphs
with tf.variable_scope('model'):
    training_outputs = createModel(input_data, len(in_vocab['vocab']), sequence_length, DA_SIZES, sum_length, layer_size=arg.layer_size)
# extract DA logits list
da_logits_list = training_outputs[:NUM_DIMS]
# summary logits at index NUM_DIMS
sum_output = training_outputs[NUM_DIMS]

# DA loss: one per dimension
# loss_dims = []
# for i in range(NUM_DIMS):
#     logits_i = da_logits_list[i]
#     labels_i = tf.reshape(das[:,:,i], [-1])  # [batch*turns]
#     ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_i, logits=logits_i)
#     ce = tf.reshape(ce, tf.shape(das)[:2])      # [batch, turns]
#     w_i = da_weights[:,:,i]
#     loss_i = tf.reduce_sum(ce * w_i, axis=1) / (tf.reduce_sum(w_i, axis=1) + 1e-12)
#     loss_dims.append(loss_i)

# Replace the DA loss section with this:
loss_dims = []
for i in range(NUM_DIMS):
    logits_i = da_logits_list[i]
    labels_i = tf.reshape(das[:,:,i], [-1])  # [batch*turns]
    
    # Apply label smoothing
    ce = sparse_label_smoothed_cross_entropy(labels_i, logits_i, 
                                           tf.reshape(da_weights[:,:,i], [-1]), 
                                           smoothing=0.05)
    
    ce = tf.reshape(ce, tf.shape(das)[:2])      # [batch, turns]
    w_i = da_weights[:,:,i]
    loss_i = tf.reduce_sum(ce * w_i, axis=1) / (tf.reduce_sum(w_i, axis=1) + 1e-12)
    loss_dims.append(loss_i)


# average across dims and batch
da_loss = tf.reduce_mean(tf.stack(loss_dims, axis=1))
params = tf.trainable_variables()
opt = tf.train.AdamOptimizer(learning_rate=0.0005)

# summary loss unchanged
if joint_training:
    with tf.variable_scope('sum_loss'):
        sum_loss = tf.contrib.seq2seq.sequence_loss(logits=sum_output, targets=summ, weights=sum_weights)

    sum_params = []
    da_params = []
    for p in params:
        if not 'da_' in p.name:
            sum_params.append(p)
        if 'da_' in p.name or 'bidirectional_rnn' in p.name or 'embedding' in p.name:
            da_params.append(p)

    gradients_da = tf.gradients(da_loss, da_params)
    gradients_sum = tf.gradients(sum_loss, sum_params)

    clipped_gradients_da, norm_da = tf.clip_by_global_norm(gradients_da, 5.0)
    clipped_gradients_sum, norm_sum = tf.clip_by_global_norm(gradients_sum, 5.0)

    gradient_norm_da = norm_da
    gradient_norm_sum = norm_sum
    update_da = opt.apply_gradients(zip(clipped_gradients_da, da_params))
    update_sum = opt.apply_gradients(zip(clipped_gradients_sum, sum_params), global_step=global_step)

    training_outputs = [global_step, da_loss, sum_loss, update_sum, update_da, gradient_norm_da, gradient_norm_sum]
else:
    # Only calculate DA loss for independent training
    total_loss = da_loss
    
    # Only update DA parameters for independent training
    da_params = [p for p in params if 'da_' in p.name or 'bidirectional_rnn' in p.name or 'embedding' in p.name]
    gradients_da = tf.gradients(da_loss, da_params)
    clipped_gradients_da, norm_da = tf.clip_by_global_norm(gradients_da, 5.0)
    gradient_norm_da = norm_da
    update_da = opt.apply_gradients(zip(clipped_gradients_da, da_params), global_step=global_step)
    
    # Create dummy operations for summary components to maintain compatibility
    dummy_sum_loss = tf.constant(0.0)
    dummy_norm_sum = tf.constant(0.0)
    dummy_update_sum = tf.no_op()
    
    training_outputs = [global_step, da_loss, dummy_sum_loss, dummy_update_sum, update_da, gradient_norm_da, dummy_norm_sum]

inputs = [input_data, sequence_length, das, da_weights, summ, sum_weights, sum_length]

# Create Inference Model
with tf.variable_scope('model', reuse=True):
    inference_outputs = createModel(input_data, len(in_vocab['vocab']), sequence_length, len(da_vocab['vocab']), sum_length, layer_size=layer_size, isTraining=False)

inference_da_output = tf.nn.softmax(inference_outputs[0], name='da_output')
inference_sum_output = tf.nn.softmax(inference_outputs[1], name='sum_output')

# Create softmax outputs for each DA dimension
inference_da_output = []
for i in range(NUM_DIMS):
    da_output = tf.nn.softmax(inference_outputs[i], name='da_output_dim_%d' % i)
    inference_da_output.append(da_output)

# Get summary sample IDs (index NUM_DIMS+1)
inference_sum_output = tf.nn.softmax(inference_outputs[NUM_DIMS], name='sum_output')

# Return all DA outputs + summary sample IDs
inference_outputs = inference_da_output + [inference_sum_output]

# inference_outputs = [inference_da_output, inference_sum_output]
inference_inputs = [input_data, sequence_length, sum_length]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

saver = tf.train.Saver()
best_sent_saver = tf.train.Saver()

if arg.evaluate_model:
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess,arg.ckpt)
    train_path = 'train'
    valid_path = 'valid'
    test_path = 'test'
    logging.info('Valid:')
    _ = valid_model(os.path.join(full_valid_path, arg.input_file), os.path.join(full_valid_path, arg.da_file), os.path.join(full_valid_path, arg.sum_file),sess, 
                                                                  save_predictions=True, prediction_prefix=os.path.join(arg.result_path + "/predictions", "valid"))
    logging.info('Test:')
    _ = valid_model(os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.da_file), os.path.join(full_test_path, arg.sum_file),sess, 
                                                                  save_predictions=True, prediction_prefix=os.path.join(arg.result_path + "/predictions", "test"))
    exit(0)

# Start Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    data_processor = None
    epochs = 0
    step = 0
    loss = 0.0
    sum_loss = 0.0
    num_loss = 0
    early_stop = arg.early_stop
    no_improve = 0

    #variables to store highest values among epochs, v stands for valid and t stands for test
    valid_prec = -1
    test_prec = -1
    valid_metrics = {}
    test_metrics = {}
    v_r1 = -1
    v_r2 = -1
    v_r3 = -1
    v_rL = -1
    t_r1 = -1
    t_r2 = -1
    t_r3 = -1
    t_rL = -1

    logging.info('Training Start')
    while True:
        if data_processor == None:
            print("DEBUG: Creating new DataProcessor")
            data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file), os.path.join(full_train_path, arg.da_file), os.path.join(full_train_path, arg.sum_file), in_vocab, is_training=True)
        
        # print("DEBUG: Getting batch")
        in_data, da_data, da_weight, length, sums,sum_weight,sum_lengths,_,_,_ = data_processor.get_batch(batch_size)
        # print("DEBUG: Batch shapes - in_data: {}, da_data: {}".format(in_data.shape, da_data.shape))
        if in_data is not None and len(in_data) > 0:
            feed_dict = {input_data.name: in_data, das.name: da_data, da_weights.name: da_weight, sequence_length.name: length, summ.name: sums, sum_weights.name: sum_weight, sum_length.name: sum_lengths}
        if data_processor.end != 1:
            #in case training data can be divided by batch_size,
            #which will produce an "empty" batch that has no data with data_processor.end==1
            # print("DEBUG: Running training step")
            ret = sess.run(training_outputs, feed_dict)
            # print("DEBUG: Training step completed")
            loss += np.mean(ret[1])
            sum_loss += np.mean(ret[2])
            step = ret[0]
            num_loss += 1

        if data_processor.end == 1:
            data_processor.close()
            data_processor = None
            epochs += 1
            logging.info('Step: ' + str(step))
            logging.info('Epochs: ' + str(epochs))
            logging.info('DA Loss: ' + str(loss/num_loss))
            logging.info('Int. Loss: ' + str(sum_loss/num_loss))
            num_loss = 0
            loss = 0.0
            sum_loss = 0.0

            save_path = os.path.join(arg.model_path, model_type)
            save_path += '_size_' + str(layer_size) + '_epochs_' + str(epochs) + '.ckpt'
            saver.save(sess, save_path)

            logging.info('Valid:')
            #variable starts wih e stands for current epoch
            e_v_r1, e_v_r2,e_v_r3,e_v_rL,e_valid_metrics = valid_model(os.path.join(full_valid_path, arg.input_file), os.path.join(full_valid_path, arg.da_file), os.path.join(full_valid_path, arg.sum_file), sess, 
                                                                  save_predictions=True, prediction_prefix=os.path.join(arg.result_path + "/predictions", "valid"))
            logging.info('Test:')
            e_t_r1, e_t_r2,e_t_r3,e_t_rL,e_test_metrics = valid_model(os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.da_file), os.path.join(full_test_path, arg.sum_file), sess, 
                                                                 save_predictions=True, prediction_prefix=os.path.join(arg.result_path + "/predictions", "test"))
            
            e_valid_prec = e_valid_metrics['micro_precision']
            e_test_prec = e_test_metrics['micro_precision']
            

            if e_v_r2 <= v_r2 and e_valid_prec <= valid_prec:
                no_improve += 1
            else:
                no_improve = 0

            if e_valid_prec > valid_prec:
                valid_prec = e_valid_prec
                test_prec = e_test_prec
                valid_metrics = e_valid_metrics
                test_metrics = e_test_metrics

            if e_v_r2 > v_r2:
                v_r2 = e_v_r2
            if e_v_r1 > v_r1:
                v_r1 = e_v_r1
            if e_v_r3 > v_r3:
                v_r3 = e_v_r3
            if e_v_rL > v_rL:
                v_rL = e_v_rL
            if e_t_r1 > t_r1:
                t_r1 = e_t_r1
            if e_t_r2 > t_r2:
                t_r2 = e_t_r2
            if e_t_r3 > t_r3:
                t_r3 = e_t_r3
            if e_t_rL > t_rL:
                t_rL = e_t_rL

                #save best model
                save_path=os.path.join(arg.model_path, 'best_sent_'+str(layer_size)+'/')+'epochs_'+str(epochs)+'.ckpt'
                best_sent_saver.save(sess,save_path)

            if epochs == arg.max_epochs:
                break

            if early_stop == True:
                if no_improve > arg.patience:
                    break

            if test_prec == -1 or valid_prec == -1 or t_r2 == -1 or v_r2 == -1:
                print('something in validation or testing goes wrong! did not update error.')
                exit(1)

header = arg.result_path
with open(os.path.join(header,'valid_da_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write("Epochs: {}\nJoint Training: {}\n".format(epochs, joint_training))
    f.write("=== FUNCTION-LEVEL METRICS (Recommended) ===\n")
    f.write("Micro-Precision: {}%\n".format(valid_metrics['micro_precision']))
    f.write("Micro-Recall: {}%\n".format(valid_metrics['micro_recall']))
    f.write("Micro-F1: {}%\n".format(valid_metrics['micro_f1']))
    f.write("Macro-Precision: {}%\n".format(valid_metrics['macro_precision']))
    f.write("Macro-Recall: {}%\n".format(valid_metrics['macro_recall']))
    f.write("Macro-F1: {}%\n".format(valid_metrics['macro_f1']))
    f.write("=== DIMENSION-LEVEL METRICS ===\n")
    f.write("Micro-Precision: {}%\n".format(valid_metrics['micro_precision_dim']))
    f.write("Micro-Recall: {}%\n".format(valid_metrics['micro_recall_dim']))
    f.write("Micro-F1: {}%\n".format(valid_metrics['micro_f1_dim']))
    f.write("Macro-Precision: {}%\n".format(valid_metrics['macro_precision_dim']))
    f.write("Macro-Recall: {}%\n".format(valid_metrics['macro_recall_dim']))
    f.write("Macro-F1: {}%\n".format(valid_metrics['macro_f1_dim']))
    f.write("=== OTHER METRICS ===\n")
    f.write("Hamming Loss: {}%\n".format(valid_metrics['hamming_loss']))
    f.write("Exact Match: {}%\n".format(valid_metrics['exact_match']))
    f.write("Average active dimensions per turn: pred={:.2f}, true={:.2f}\n".format(valid_metrics['avg_active_pred'], valid_metrics['avg_active_true']))
    f.write("Function-Level Total: TP: {}, FP: {}, FN: {}\n".format(
        valid_metrics['function_level_metrics']['total_tp'],
        valid_metrics['function_level_metrics']['total_fp'],
        valid_metrics['function_level_metrics']['total_fn']
    ))
    f.write("Dimension-Level Total: TP: {}, FP: {}, FN: {}\n".format(
        valid_metrics['dimension_level_metrics']['total_tp'],
        valid_metrics['dimension_level_metrics']['total_fp'],
        valid_metrics['dimension_level_metrics']['total_fn']
    ))
    f.write("Per-Dimension Accuracy:\n")
    for dim_result in valid_metrics['dimension_counts_formatted']:
        f.write("  {}\n".format(dim_result))
    f.write("Per-Function Accuracy:\n")
    for func_result in valid_metrics['function_counts_formatted']:
        f.write("  {}\n".format(func_result))
    f.write("="*50 + "\n")
with open(os.path.join(header,'test_da_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write("Epochs: {}\nJoint Training: {}\n".format(epochs, joint_training))
    f.write("=== FUNCTION-LEVEL METRICS (Recommended) ===\n")
    f.write("Micro-Precision: {}%\n".format(test_metrics['micro_precision']))
    f.write("Micro-Recall: {}%\n".format(test_metrics['micro_recall']))
    f.write("Micro-F1: {}%\n".format(test_metrics['micro_f1']))
    f.write("Macro-Precision: {}%\n".format(test_metrics['macro_precision']))
    f.write("Macro-Recall: {}%\n".format(test_metrics['macro_recall']))
    f.write("Macro-F1: {}%\n".format(test_metrics['macro_f1']))
    f.write("=== DIMENSION-LEVEL METRICS ===\n")
    f.write("Micro-Precision: {}%\n".format(test_metrics['micro_precision_dim']))
    f.write("Micro-Recall: {}%\n".format(test_metrics['micro_recall_dim']))
    f.write("Micro-F1: {}%\n".format(test_metrics['micro_f1_dim']))
    f.write("Macro-Precision: {}%\n".format(test_metrics['macro_precision_dim']))
    f.write("Macro-Recall: {}%\n".format(test_metrics['macro_recall_dim']))
    f.write("Macro-F1: {}%\n".format(test_metrics['macro_f1_dim']))
    f.write("=== OTHER METRICS ===\n")
    f.write("Hamming Loss: {}%\n".format(test_metrics['hamming_loss']))
    f.write("Exact Match: {}%\n".format(test_metrics['exact_match']))
    f.write("Average active dimensions per turn: pred={:.2f}, true={:.2f}\n".format(test_metrics['avg_active_pred'], test_metrics['avg_active_true']))
    f.write("Function-Level Total: TP: {}, FP: {}, FN: {}\n".format(
        test_metrics['function_level_metrics']['total_tp'],
        test_metrics['function_level_metrics']['total_fp'],
        test_metrics['function_level_metrics']['total_fn']
    ))
    f.write("Dimension-Level Total: TP: {}, FP: {}, FN: {}\n".format(
        test_metrics['dimension_level_metrics']['total_tp'],
        test_metrics['dimension_level_metrics']['total_fp'],
        test_metrics['dimension_level_metrics']['total_fn']
    ))
    f.write("Per-Dimension Accuracy:\n")
    for dim_result in test_metrics['dimension_counts_formatted']:
        f.write("  {}\n".format(dim_result))
    f.write("Per-Function Accuracy:\n")
    for func_result in test_metrics['function_counts_formatted']:
        f.write("  {}\n".format(func_result))
    f.write("="*50 + "\n")
with open(os.path.join(header,'valid_r1_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(v_r1)+'\n')
with open(os.path.join(header,'test_r1_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(t_r1)+'\n')
with open(os.path.join(header,'valid_r2_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(v_r2)+'\n')
with open(os.path.join(header,'test_r2_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(t_r2)+'\n')
with open(os.path.join(header,'valid_r3_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(v_r3)+'\n')
with open(os.path.join(header,'test_r3_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(t_r3)+'\n')
with open(os.path.join(header,'valid_rL_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(v_rL)+'\n')
with open(os.path.join(header,'test_rL_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(t_rL)+'\n')

print('*'*20+model_type+' '+str(layer_size)+'*'*20)
