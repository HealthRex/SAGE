from __future__ import print_function

import os
import tensorflow as tf     
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import random
import numpy as np
import csv
import pdb
from sklearn import metrics
import pandas as pd
import tensorflow_addons as tfa
from sklearn.metrics import roc_auc_score

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def convert_1d_2d(current_batch                                                                                        
                , num_time_steps
                , d_data
                , n_targets):
    one=1
    # pdb.set_trace()
    current_batch_ar=np.array(current_batch)
    current_batch_x = current_batch_ar[:,:-n_targets-1]
    current_batch_y = current_batch_ar[:,-n_targets-1:-1]

    current_batch_length=[]
    for i in range(len(current_batch_x)):
        current_batch_length.append(find_length(current_batch_x[i]))

    current_batch_x_reshaped=np.reshape(current_batch_x,(len(current_batch_x), num_time_steps, d_data+1))  # +1 for visit date 


    return current_batch_x_reshaped[:,:,one:], current_batch_y, current_batch_length


def read_a_batch(file, batch_size):
    # pdb.set_trace()
    line_counter = 1
    eof_reached = 0
    current_line = file.readline()
    if current_line.split(',')[0] == 'Patient_ID':
       current_line = file.readline() 
    # If it is EOF start from beginning
    if current_line == '':
        eof_reached = 1
        file.seek(0)
        current_line = file.readline()
        if current_line.split(',')[0] == 'Patient_ID':
            current_line = file.readline() 
    current_line = current_line.split(',')
    current_line = [float(i) for i in current_line[1:]]
    current_batch = [] 
    current_batch.append(current_line)   
    while line_counter < batch_size:
        current_line = file.readline()
        if current_line.split(',')[0] == 'Patient_ID':
            current_line = file.readline() 
        if current_line == '':
            eof_reached = 1
            file.seek(0)        
            current_line = file.readline() 
            if current_line.split(',')[0] == 'Patient_ID':
                current_line = file.readline() 
        current_line = current_line.split(',')
        current_line = [float(i) for i in current_line[1:]]        
        current_batch.append(current_line)   
        line_counter += 1
    return current_batch, eof_reached




def dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden, drp):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # pdb.set_trace()
    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.keras.layers.LSTMCell(units=n_hidden, dropout = drp, kernel_regularizer='l2')#, activation='gelu')
    # lstm_cell = tfa.rnn.LayerNormLSTMCell(units=n_hidden, dropout = drp, kernel_regularizer='l2')
    # lstm_cell = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_cell)#,input_keep_prob=0.5, output_keep_prob=keep_prob)


    # Get lstm cell output, providing 'sequence_length' will perform dynamic
    # calculation.
    lstm_input = x
    outputs, states = tf.compat.v1.nn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    outputs_original=outputs
    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen-1)
    # Indexing
    output_before_idx =outputs
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    output_after_idx =outputs
    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out'], states, outputs, outputs_original, output_before_idx, output_after_idx, lstm_input

# Parameters
def find_length(sequence):
    # pdb.set_trace()
    sequence = np.reshape(sequence, (252, 301))
    length=0
    for i in range(len(sequence)):
        if sum(sequence[i][1:]) != 0:
            length=i
    return (length+1)
  
def main(idx
    , drp
    , epochs
    , reg_coeff
    , learning_rt
    , n_hid
    , batch_sz
    , train_filename
    , test_filename
    , n_targets):
    tf.compat.v1.reset_default_graph()
    # pdb.set_trace()
    print("Creating and training the single stream LSTM model ...")
    learning_rate = learning_rt
    training_iters_low = 10000
    batch_size = batch_sz
    display_step = 10
    loss_threshold = 0.0001
    n_hidden = n_hid 
    # num_classes = 2 
    num_time_steps=252
    seq_max_len = 252
    d_data =  300
    one = 1
    zero = 0
    accuracies=[]

    test_data = pd.read_csv(test_filename, skiprows=1, header=None)#, nrows=10)
    test_data_predictors = np.reshape(test_data.iloc[:,1:-n_targets-1].values, (len(test_data),num_time_steps,d_data+1))
    test_data_x = test_data_predictors[:,:,1:]  
    test_data_y = test_data.iloc[:,-n_targets-1:-1]
    
    test_length=[]
    for i in range(len(test_data)):
        test_length.append(find_length(test_data.iloc[:,1:-n_targets-1].values[i]))
        


# tf Graph input
    tf.compat.v1.disable_eager_execution()
    x = tf.compat.v1.placeholder("float", [None, num_time_steps, d_data ])  # input sequence
    y = tf.compat.v1.placeholder("float", [None, n_targets])       # labels
# A placeholder for indicating each sequence length
    seqlen = tf.compat.v1.placeholder(tf.int32, [None])               # sequence length

# Define weights
    weights = {
        'out': tf.Variable(tf.random.normal([n_hidden, n_targets]))
    }
    biases = {
        'out': tf.Variable(tf.random.normal([n_targets]))
    }


    pred, states, outputs, outputs_original,output_before_idx, output_after_idx, lstm_input = dynamicRNN(x, seqlen, weights, biases,seq_max_len,n_hidden, drp)

    
# Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)) #+  (reg_coeff * tf.reduce_sum(tf.nn.l2_loss(weights['out'])))
    
    # optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
    # correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # pred_arg=tf.argmax(pred,1)
    # y_arg=tf.argmax(y,1)
    sigmoid_predictions = tf.nn.sigmoid(pred)

# Initializing the variables
    init = tf.compat.v1.global_variables_initializer()

    saver = tf.compat.v1.train.Saver()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(init)
        
        num_passed_epochs = 0           
        #with open(train_meds_filename) as train_meds_file, open(train_diags_filename) as train_diags_file, open(train_procs_filename) as train_procs_file, open(train_demogs_filename) as train_demogs_file, open(valid_meds_filename) as valid_meds_file, open(valid_diags_filename) as valid_diags_file, open(valid_procs_filename) as valid_procs_file, open(valid_demogs_filename) as valid_demogs_file, open ('results/LSTM_single_stream/Loss_train_lstm_single_stream_'+str(idx)+'.csv', 'w') as loss_file , open('results/LSTM_single_stream/Loss_test_lstm_single_stream_'+str(idx)+'.csv', 'w') as loss_file_val:
        train_file = open(train_filename)
        
        train_loss_file = open('results/LSTM/Loss_train_lstm_'+str(idx)+'.csv', 'w')        
        # test_loss_file = open('results/LSTM/Loss_test_lstm_'+str(idx)+'.csv', 'w')
        
        step=0

        while num_passed_epochs < epochs:
            # num_passed_epochs+=1 ###########
            # pdb.set_trace()
            current_batch, eof_reached = read_a_batch(train_file, batch_size)

            if  eof_reached == 1:
                # pdb.set_trace()
                num_passed_epochs += 1
                print(num_passed_epochs)
               
            current_batch_final, current_batch_final_labels, current_batch_length = convert_1d_2d(current_batch                                                                                        
                                                                                        , num_time_steps
                                                                                        , d_data
                                                                                        , n_targets)            


            # pdb.set_trace()
            # data_temp = pd.read_csv(train_file, skiprows=1, header=None, nrows=40)
            sess.run(optimizer, feed_dict={x: current_batch_final, y: current_batch_final_labels, seqlen: current_batch_length})  
            # [pred, states, outputs, outputs_original,output_before_idx, output_after_idx, lstm_input] = sess.run([pred, states, outputs, outputs_original,output_before_idx, output_after_idx, lstm_input], feed_dict={x: current_batch_final, y: current_batch_final_labels, seqlen: current_batch_length})  
            loss = sess.run(cost, feed_dict={x: current_batch_final, y: current_batch_final_labels, seqlen: current_batch_length})
            
            train_loss_file.write(str(loss))
            train_loss_file.write("\n")
            step+=1
            # print(step)
            # if step>5:
            #     break
            if step%10==0:
                print('The step is {} and the epoch number is {}'.format(step, num_passed_epochs))                
                saver.save(sess, save_path='saved_deep_learning_models/LSTM/LSTM_model_'+str(int(idx))+'.ckpt')
     
        
        print("Optimization Finished!")
        train_file.close()
        train_loss_file.close()
        # test_loss_file.close()
        # pdb.set_trace()
        #=================================================================
        # pdb.set_trace()
        # ==== Validating

        [sigmoid_predictions_temp] =sess.run([sigmoid_predictions], feed_dict={x: test_data_x, y: test_data_y.values, seqlen: test_length})
        test_auc_micro = roc_auc_score(test_data_y.values, sigmoid_predictions_temp, multi_class='ovr', average='micro')

        print("=================================")
        predictions = []

        for i in range(len(sigmoid_predictions_temp)):
            predictions.append([1 if x>0.5 else 0 for x in sigmoid_predictions_temp[i]])
        # pdb.set_trace()    
        print('test') 
        results_report = metrics.classification_report(y_true=test_data_y.values, y_pred=np.array(predictions), output_dict=True)
        results_report_df = pd.DataFrame(results_report).transpose()
               
    return sigmoid_predictions_temp, results_report_df, test_auc_micro

if __name__ == "__main__": main(idx, drp, epochs, reg_coeff, learning_rt, n_hid, batch_sz, train_filename, test_filename, n_targets)


