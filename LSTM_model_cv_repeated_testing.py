import  models.LSTM_repeated_testing as lstm
import os
import random as rnd
import pdb
import numpy as np
import random
import time
import pdb
import pandas as pd
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
num_time_steps = 252
d_data =  300
# pdb.set_trace()
start_time = time.time()
train_filename = 'none'

test_filename = 'longitudinal_data/for_recommender_longitudinal_test.csv'

# pdb.set_trace()

test_data = pd.read_csv(test_filename, skiprows=1, header=None)#, nrows=100)

columns = test_data.columns
columns = [str(x) for x in columns]
columns[0] = 'Patient_ID'

test_data.columns = columns
with open('results/LSTM/repeated_test/results_LSTM_recommender_repeated_testing.csv', 'w') as res_file:
    res_file.write('Micro avg precision, Micro avg recall, Micro avg F1, Micro avg AUC\n')
    
    for i in range(10):
        print(i)
        current_test_fold_stationary = pd.read_csv('saved_classical_ml_models/kfolds_testing_recommender/test_fold'+str(i)+'_recommender.csv')
        current_test_fold_ids = current_test_fold_stationary['Patient_ID'].values
        
        current_test_fold = test_data[test_data['Patient_ID'].isin(current_test_fold_ids)]

        # current_test_fold = test_data

        sigmoid_predictions_temp, results_report_df, test_auc_micro =lstm.main(model_number
                                                                            , dropout_r
                                                                            , num_epochs
                                                                            , regu_fact
                                                                            , learning_r
                                                                            , n_hid 
                                                                            , batch_sz
                                                                            , current_test_fold
                                                                            , n_targets)
        
        # pdb.set_trace()
        np.savetxt('results/LSTM/repeated_test/sigmoid_predictions_lstm_'+str(i)+'.csv', sigmoid_predictions_temp, delimiter=',')
        results_report_df.to_csv('results/LSTM/repeated_test/results_LSTM_recommender_'+str(i)+'.csv')

        res_file.write(str(results_report_df['precision'].loc['micro avg']))
        res_file.write(',')

        res_file.write(str(results_report_df['recall'].loc['micro avg']))
        res_file.write(',')

        res_file.write(str(results_report_df['f1-score'].loc['micro avg']))
        res_file.write(',')
        
        res_file.write(str(test_auc_micro))
        res_file.write('\n')
