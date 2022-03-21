import pdb
import argparse
import sys
import os
import models.ml_recommender_models as cl_ml


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()  


# === train, validation and test stationary data
parser.add_argument("--train_data_path", type=str, default='recommender_data/recomender_data_train.csv')    
parser.add_argument("--test_data_path", type=str, default='recommender_data/recomender_data_test.csv')    

parser.add_argument("--ml_model", type=str, default='none') 

parser.add_argument("--fs_flag", type=int, default=0, choices=[0, 1])    
parser.add_argument("--fs_method", type=str, default='rf') 
parser.add_argument("--top_n_features", type=int, default=30) 
parser.add_argument("--repeated_test", type=int, default=0, choices=[0, 1])    

parser.add_argument("--trained_rf_path", type=str, default= 'saved_classical_ml_models/rf_model_recommender.pkl')    
parser.add_argument("--trained_lr_path", type=str, default= 'saved_classical_ml_models/lr_model_recommender.pkl')    
parser.add_argument("--trained_xgb_path", type=str, default= 'saved_classical_ml_models/xgb_model_recommender.pkl')    


args = parser.parse_args()
if  args.ml_model == 'rf' and args.fs_flag==1:
    print('Performing feature selection first')
    # pdb.set_trace()
    path_to_features, dropped_targets = cl_ml.rf_feature_selection(args.train_data_path
                                                , args.test_data_path
                                                , args.num_target)
    print('Starting to train a random forest model using:\n')
    cl_ml.random_forest_model(args.train_data_path
                            ,args.test_data_path
                            ,path_to_features
                            ,args.top_n_features
                            , args.num_target
                            , dropped_targets)

elif  args.ml_model == 'rf' and args.fs_flag==0:
    print('Performing feature selection first')
    print('Starting to train a random forest model using:\n')
    cl_ml.random_forest_model(args.train_data_path
                            ,args.test_data_path
                            ,'NONE'
                            ,args.top_n_features
                            , 'NONE')

elif  args.ml_model == 'xgb' and args.fs_flag==0:
    print('Performing feature selection first')
    print('Starting to train a random forest model using:\n')
    cl_ml.xgboost_model(args.train_data_path
                            ,args.test_data_path
                            ,'NONE'
                            ,args.top_n_features
                            , 'NONE')
elif  args.ml_model == 'bl' and args.fs_flag==0:
    print('Performing feature selection first')
    print('Starting to train a random forest model using:\n')
    cl_ml.baseline_models(args.test_data_path)

elif args.ml_model == 'none' and args.fs_flag==0:
    print('No ML model has been selected')

if args.repeated_test == 1:
    cl_ml.repeated_test(args.trained_rf_path
                        , args.trained_xgb_path
                        , args.test_data_path
                        )


