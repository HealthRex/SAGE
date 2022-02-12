import pdb
import argparse
import sys
import os
import models.ml_models as cl_ml


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()  


# === train, validation and test stationary data
parser.add_argument("--train_data_path", type=str, default='stationary_data/stationary_data_imbratio1_normalized_train.csv')    
parser.add_argument("--test_data_path", type=str, default='stationary_data/stationary_data_imbratio1_normalized_test.csv')    

parser.add_argument("--ml_model", type=str, default='rf') 


args = parser.parse_args()
if  args.ml_model == 'rf':
    print('Starting to train a random forest model using:\n')
    cl_ml.random_forest_model(args.train_data_path
                            ,args.test_data_path)
