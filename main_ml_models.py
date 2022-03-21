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
parser.add_argument("--stationary_mci_path", type=str, default='stationary_data/stationary_dataset_mci.csv')    
parser.add_argument("--stationary_nonmci_path", type=str, default='stationary_data/stationary_dataset_nonmci.csv')    


parser.add_argument("--path_to_features", type=str, default='saved_classical_ml_models/feature_impoerance_rf.csv')    
parser.add_argument("--trained_rf_path", type=str, default= 'saved_classical_ml_models/rf_model.pkl')    
parser.add_argument("--trained_lr_path", type=str, default= 'saved_classical_ml_models/lr_model.pkl')    
parser.add_argument("--trained_xgb_path", type=str, default= 'saved_classical_ml_models/xgb_model.pkl')    


parser.add_argument("--ml_model", type=str, default='none', choices=['rf', 'lr', 'xgb']) 

parser.add_argument("--fs_flag", type=int, default=0, choices=[0, 1])    
parser.add_argument("--imb_test", type=int, default=0, choices=[0, 1])    
parser.add_argument("--fs_method", type=str, default='rf') 
parser.add_argument("--top_n_features", type=int, default=30) 


args = parser.parse_args()
if  args.fs_flag==1:
    print('Performing feature selection first')
    cl_ml.rf_feature_selection(args.train_data_path)


if  args.ml_model == 'rf':
    print('Starting to train a random forest model using:\n')
    cl_ml.random_forest_model(args.train_data_path
                            ,args.test_data_path
                            ,args.path_to_features
                            ,args.top_n_features)

elif  args.ml_model == 'lr':
    print('Starting to train a logistic regression model using:\n')
    cl_ml.logistic_regression_model(args.train_data_path
                            ,args.test_data_path
                            ,args.path_to_features
                            ,args.top_n_features)

elif  args.ml_model == 'xgb':
    print('Starting to train a logistic regression model using:\n')
    cl_ml.xgboost_model(args.train_data_path
                            ,args.test_data_path
                            ,args.path_to_features
                            ,args.top_n_features)    
else:
    print('No ML model has been selected ... ')


if args.imb_test == 1:
    cl_ml.test_with_imb(args.trained_rf_path
                        , args.trained_lr_path
                        , args.trained_xgb_path
                        , args.train_data_path
                        , args.test_data_path
                        # , args.stationary_mci_path
                        , args.stationary_nonmci_path
                        , args.path_to_features
                        , args.top_n_features
                        )


