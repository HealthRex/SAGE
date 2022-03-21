import  models.LSTM as lstm
import os
import random as rnd
import pdb
import numpy as np
import random
import time
import pdb
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


# pdb.set_trace()
model_number = 1000
dropout_r = 0.4
num_epochs = 25
regu_fact = 0.000001
learning_r = 0.01
n_hid = 128
batch_sz = 256
n_targets = 46

# pdb.set_trace()
start_time = time.time()
train_filename = 'longitudinal_data/for_recommender_longitudinal_train.csv'

test_filename = 'longitudinal_data/for_recommender_longitudinal_test.csv'

print('Concatenating train and validation files ...')

print('========================')
print('Train and test file names are:')
print(train_filename)
print(test_filename)

sigmoid_predictions_temp, results_report_df, test_auc_micro =lstm.main(model_number
                                                                    , dropout_r
                                                                    , num_epochs
                                                                    , regu_fact
                                                                    , learning_r
                                                                    , n_hid 
                                                                    , batch_sz
                                                                    , train_filename
                                                                    , test_filename
                                                                    , n_targets)
# pdb.set_trace()
np.savetxt('results/LSTM/sigmoid_predictions_lstm.csv', sigmoid_predictions_temp, delimiter=',')
results_report_df.to_csv('results/LSTM/results_LSTM_recommender.csv')

with open('results/LSTM/results_LSTM_recommender_detailed_auc.csv', 'w') as auc_file:
    auc_file.write('Micro auc is:\n')
    auc_file.write(str(test_auc_micro))
    auc_file.write('\n')
