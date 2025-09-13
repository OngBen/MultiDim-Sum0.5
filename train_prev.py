import os
import argparse
import logging
import sys
import shutil
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import rnn_cell_impl

from utils_prev import loadSentenceVocabulary
from utils_prev import loadDAVocabulary
from utils_prev import computeAccuracy
from utils_prev import DataProcessor
from utils_prev import DA_ISO_MAPPING
import rouge

parser = argparse.ArgumentParser(allow_abbrev=False)

#Network
parser.add_argument("--num_units", type=int, default=256, help="Network size.", dest='layer_size')
parser.add_argument("--model_type", type=str, default='full', help="""full | summary_only(default)
                                                                    full: full attention model
                                                                    summary_only: summary attention model""")

#Training Environment
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--max_epochs", type=int, default=30, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false',dest='early_stop', help="Disable early stop, which is based on dialogue act accuracy and ROUGE-2 of summary.")
parser.add_argument("--patience", type=int, default=10, help="Patience to wait before stop.")

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
parser.add_argument("--da_file", type=str, default='da', help="suffix of dialogue act label file.")
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
in_vocab = loadSentenceVocabulary(os.path.join(vocab_path, 'in_vocab'))
da_vocab = loadDAVocabulary(os.path.join(vocab_path, 'da_vocab'))

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

    with tf.variable_scope('sentence_gated'):
        sum_gate = rnn_cell_impl._linear(sum_output, attn_size, True)
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

        # Preserve 3D shape [batch, seq_len, attn_size]
        batch_size = tf.shape(sentence_gate)[0]
        seq_len = tf.shape(sentence_gate)[1]
        sentence_gate = tf.reshape(sentence_gate, [batch_size, seq_len, -1])

        # Ensure da_inputs is 3D (e.g., [batch, seq_len, features])
        da_output = tf.concat([sentence_gate, da_inputs], axis=2)

    with tf.variable_scope('da_proj'):
        # Reshape to 2D for linear layer, then back to 3D
        da_output_2d = tf.reshape(da_output, [-1, da_output.shape[2]])
        da_logits_2d = rnn_cell_impl._linear(da_output_2d, len(da_vocab['vocab']), True)
        da_logits = tf.reshape(da_logits_2d, [batch_size, seq_len, len(da_vocab['vocab'])])


    outputs = [da_logits, decoder_final_outputs.rnn_output, decoder_final_outputs.sample_id]
    #print("DA: ",da)
    return outputs

def valid_model(in_path, da_path, sum_path,sess):
    #return accuracy for dialogue act, rouge-1,2,3,L for summary
    #some useful items are also calculated
    #da_outputs, correct_das: predicted / ground truth of dialogue act

    rouge_1 = []
    rouge_2 = []
    rouge_3 = []
    rouge_L = []
    da_outputs = []
    correct_das = []

    data_processor_valid = DataProcessor(in_path, da_path, sum_path, in_vocab, da_vocab)

    dim_slices = {}
    current_idx = 0
    for dim in da_dimensions:
        dim_size = len(da_functions[dim])
        dim_slices[dim] = slice(current_idx, current_idx + dim_size)
        current_idx += dim_size

    print(da_vocab['rev'])
    
    while True:
        #get a batch of data
        in_data, da_data, da_weight, length, sums, sum_weight,sum_lengths, in_seq, da_seq, sum_seq = data_processor_valid.get_batch(batch_size)
        feed_dict = {input_data.name: in_data, sequence_length.name: length, sum_length.name: sum_lengths}
        if data_processor_valid.end != 1 or in_data:
            ret = sess.run(inference_outputs, feed_dict)

            #summary part
            pred_sums = []
            correct_sums = []
            for batch in ret[1]:
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

            #dialogue act part
            pred_das = ret[0].reshape((da_data.shape[0], da_data.shape[1], -1))
            #print(pred_das)
            #print(da_data)
            for p, t, i, l in zip(pred_das, da_data, in_data, length):
                p = np.argmax(p, 1)
                tmp_pred = []
                tmp_correct = []
                for j in range(l):
                    tmp_pred.append(da_vocab['rev'][p[j]])
                    tmp_correct.append(da_vocab['rev'][t[j]])
                da_outputs.append(tmp_pred)
                correct_das.append(tmp_correct)

        if data_processor_valid.end == 1:
            break

    print(da_outputs)
    print(correct_das)

    precision = computeAccuracy(correct_das, da_outputs)
    logging.info('da precision: ' + str(precision))
    logging.info('sum rouge1: ' + str(np.mean(rouge_1)))
    logging.info('sum rouge2: ' + str(np.mean(rouge_2)))
    logging.info('sum rouge3: ' + str(np.mean(rouge_3)))
    logging.info('sum rougeL: ' + str(np.mean(rouge_L)))

    data_processor_valid.close()
    return np.mean(rouge_1),np.mean(rouge_2),np.mean(rouge_3),np.mean(rouge_L),precision

#Start of training
da_vocab = loadDAVocabulary(os.path.join(arg.vocab_path, 'da_vocab'))
da_vocab_size = len(da_vocab['vocab'])
da_dimensions = da_vocab['dimensions']
da_functions = da_vocab['dim_func_map']

print('Dialogue Act Dimensions:', da_dimensions)
print('Dialogue Acts per Dimension:', da_functions)

# Create Training Model
input_data = tf.placeholder(tf.int32, [None, None, None], name='inputs')
sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
global_step = tf.Variable(0, trainable=False, name='global_step')
das = tf.placeholder(tf.float32, [None, None, da_vocab_size], name='das')
da_weights = tf.placeholder(tf.float32, [None, None], name='da_weights')  # Keep as 2D (batch, time)
summ = tf.placeholder(tf.int32, [None, None], name='summ')
sum_weights = tf.placeholder(tf.float32, [None, None], name='sum_weights')
sum_length = tf.placeholder(tf.int32, [None], name='sum_length')

with tf.variable_scope('model'):
    training_outputs = createModel(input_data, len(in_vocab['vocab']), sequence_length, len(da_vocab['vocab']), sum_length, layer_size=layer_size)

print("das:", das.get_shape().as_list())
print("training_outputs:", training_outputs[0].get_shape().as_list())
print("Number of DA dimensions:", len(da_dimensions))
print("DA functions lengths:", {dim: len(funcs) for dim, funcs in da_functions.items()})
# Simplified DA loss

with tf.variable_scope('da_loss'):
    total_loss = 0.0
    total_weights = 0.0
    start_idx = 0
    da_logits = training_outputs[0]
    for dim in da_vocab['dimensions']:
        dim_size = len(da_vocab['dim_func_map'][dim])
        # Slice logits and labels
        dim_logits = da_logits[:, :, start_idx:start_idx+dim_size]
        dim_labels = das[:, :, start_idx:start_idx+dim_size]
        
        # Mask padding and compute loss
        crossent = tf.nn.softmax_cross_entropy_with_logits(
            labels=dim_labels, 
            logits=dim_logits
        )
        mask = tf.reduce_sum(dim_labels, axis=-1)  # 1 where dimension is active
        total_loss += tf.reduce_sum(crossent * mask)
        total_weights += tf.reduce_sum(mask)
        start_idx += dim_size
    
    da_loss = total_loss / (total_weights + 1e-12) 

sum_output = training_outputs[1]
with tf.variable_scope('sum_loss'):
    sum_loss = tf.contrib.seq2seq.sequence_loss(logits=sum_output,targets=summ,weights=sum_weights,average_across_timesteps=False)

params = tf.trainable_variables()
opt = tf.train.AdamOptimizer(learning_rate=0.0005)

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
inputs = [input_data, sequence_length, das, da_weights, summ, sum_weights, sum_length]

# Create Inference Model
with tf.variable_scope('model', reuse=True):
    inference_outputs = createModel(input_data, len(in_vocab['vocab']), sequence_length, len(da_vocab['vocab']), sum_length, layer_size=layer_size, isTraining=False)

inference_da_output = tf.nn.softmax(inference_outputs[0], name='da_output')
inference_sum_output = tf.nn.softmax(inference_outputs[1], name='sum_output')

inference_outputs = [inference_da_output, inference_sum_output]
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
    _ = valid_model(os.path.join(full_valid_path, arg.input_file), os.path.join(full_valid_path, arg.da_file), os.path.join(full_valid_path, arg.sum_file),sess)
    logging.info('Test:')
    _ = valid_model(os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.da_file), os.path.join(full_test_path, arg.sum_file),sess)
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
    valid_da = -1
    test_da = -1
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
            data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file), os.path.join(full_train_path, arg.da_file), os.path.join(full_train_path, arg.sum_file), in_vocab, da_vocab)
        in_data, da_data, da_weight, length, sums,sum_weight,sum_lengths,_,_,_ = data_processor.get_batch(batch_size)
        feed_dict = {input_data.name: in_data, das.name: da_data, da_weights.name: da_weight, sequence_length.name: length, summ.name: sums, sum_weights.name: sum_weight, sum_length.name: sum_lengths}
        if data_processor.end != 1 or in_data:
            #in case training data can be divided by batch_size,
            #which will produce an "empty" batch that has no data with data_processor.end==1
            ret = sess.run(training_outputs, feed_dict)
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
            e_v_r1, e_v_r2,e_v_r3,e_v_rL,e_valid_da = valid_model(os.path.join(full_valid_path, arg.input_file), os.path.join(full_valid_path, arg.da_file), os.path.join(full_valid_path, arg.sum_file), sess)
            logging.info('Test:')
            e_t_r1, e_t_r2,e_t_r3,e_t_rL,e_test_da = valid_model(os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.da_file), os.path.join(full_test_path, arg.sum_file), sess)

            if e_v_r2 <= v_r2 and e_valid_da <= valid_da:
                no_improve += 1
            else:
                no_improve = 0

            if e_valid_da > valid_da:
                valid_da = e_valid_da
                test_da = e_test_da

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

            if test_da == -1 or valid_da == -1 or t_r2 == -1 or v_r2 == -1:
                print('something in validation or testing goes wrong! did not update error.')
                exit(1)

header = arg.result_path
with open(os.path.join(header,'valid_da_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(valid_da)+'\n')
with open(os.path.join(header,'test_da_'+model_type+str(layer_size)+'.txt'),'a') as f:
    f.write(str(test_da)+'\n')

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
