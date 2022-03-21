# from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import pdb
# from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
# from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import csv
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import copy
from sklearn.utils import shuffle
import random
import pickle
import matplotlib.pyplot as plt
# import shap
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier



def random_forest_model(train_data_path
                        ,test_data_path
                        ,path_to_features
                        ,top_n_features
                        # ,num_target
                        ,dropped_targets):

    # pdb.set_trace()
   
    # num_target = num_target - len(dropped_targets)
    # with open('results/classical_ml_models/rf_model.pkl', 'rb') as f:
    #     clf2 = pickle.load(f)
    print('Reading the data:')
    print(train_data_path)
    print(test_data_path)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    train_data = train_data.sample(frac=1).reset_index(drop=True)  
    test_data = test_data.sample(frac=1).reset_index(drop=True) 

    target_col_names = [x for x in train_data.columns if 'target_proc_' in x]
    
    train_data_x = train_data[train_data.columns[~train_data.columns.isin(target_col_names + ['Patient_ID'])]]
    train_data_y = train_data[train_data.columns[train_data.columns.isin(target_col_names)]]

    test_data_x = test_data[test_data.columns[~test_data.columns.isin(target_col_names + ['Patient_ID'])]]
    test_data_y = test_data[test_data.columns[test_data.columns.isin(target_col_names)]]
    # feature_ranking = pd.read_csv(path_to_features, names=['Feature', 'Score']).sort_values(by='Score', ascending=False)
    # selected_features = feature_ranking.iloc[:top_n_features, 0].values.tolist()
    # selected_features = ['Patient_ID'] + selected_features + train_data.columns.tolist()[-num_target:]
    # train_data = train_data[selected_features]
    # test_data = test_data[selected_features]

    # fig = feature_ranking.nlargest(top_n_features, columns='Score').plot(kind='barh', grid=True,figsize=(12,10))
    # fig.set_xlabel("Importance Score")
    # fig.set_ylabel("Features")
    # fig.get_figure().savefig("results/visualization_results/feature_importance_rf_recommender.png", dpi=300)
    # # fig.close()
    print('Finished reading data...')
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']#, 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]#[4, 8, 16, 32]#, 64, 128]# [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]# [4, 8, 16, 32]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]#[4, 8, 16, 32]
    # Method of selecting samples for training each tree
    # Because the data set is too large, I allways bootsrap. If bootsrap=False, the whole data set is used to train each tree
    #bootstrap = [True, False]

    # ccp_alpha = [0, 0.00001, 0.0001, 0.001, 0.1, 1]
    
    # Create the random grid
    hyperparameters = {'n_estimators': n_estimators
                   ,'max_features': max_features
                   , 'max_depth': max_depth
                   , 'min_samples_split': min_samples_split
                   , 'min_samples_leaf': min_samples_leaf
                   #, 'bootstrap': bootstrap
                   }#,'ccp_alpha': ccp_alpha}
    pdb.set_trace() 
    print('Hyperparameters:')
    print(hyperparameters)
    with open('saved_classical_ml_models/rf_hyperparameters_recommender.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in hyperparameters.items():
           writer.writerow([key, value])
    # pdb.set_trace()  
 

    randomCV = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=-1, warm_start=True, verbose=1), param_distributions=hyperparameters, n_iter=20, cv=5,scoring="roc_auc")
    randomCV.fit(train_data_x, train_data_y)
    
    train_data_x.to_csv('saved_classical_ml_models/train_predictors_recommender_final.csv', index=False)
    train_data_y.to_csv('saved_classical_ml_models/train_targets_recommender_final.csv', index=False)
    # === Save models
    with open('saved_classical_ml_models/rf_model_recommender.pkl','wb') as f:
        pickle.dump(randomCV,f)
    
    (pd.DataFrame.from_dict(data=randomCV.best_params_, orient='index').to_csv('saved_classical_ml_models/best_params_rf_recommender.csv', header=False))
    best_rf_model = randomCV.best_estimator_

    rf_predictions = best_rf_model.predict(test_data_x)    
    test_data_x.to_csv('saved_classical_ml_models/test_predictors_recommender_final.csv', index=False)
    test_data_y.to_csv('saved_classical_ml_models/test_targets_recommender_final.csv', index=False)

    np.savetxt('saved_classical_ml_models/predictions_rf_recommender.csv', rf_predictions, delimiter=',')

    results_report = metrics.classification_report(y_true=test_data_y, y_pred=np.array(rf_predictions), output_dict=True, target_names=test_data_y.columns)
   
    results_report_df = pd.DataFrame(results_report).transpose()
    results_report_df.to_csv('results/classical_ml_models/results_rf_recommender.csv')

    probabilities_test = best_rf_model.predict_proba(test_data_x)
    probabilities_test_reshaped = np.array(probabilities_test)[:,:,1].T

    rf_test_auc_micro = roc_auc_score(test_data_y, probabilities_test_reshaped, multi_class='ovr', average='micro')
    rf_test_auc_macro = roc_auc_score(test_data_y, probabilities_test_reshaped, multi_class='ovr', average='macro')
    rf_test_auc_weighted = roc_auc_score(test_data_y, probabilities_test_reshaped, multi_class='ovr', average='weighted')

    rf_test_auc_class_based = roc_auc_score(test_data_y, probabilities_test_reshaped, multi_class='ovr', average=None)

    with open('results/classical_ml_models/results_rf_recommender_detailed_auc.csv', 'w') as auc_file:
        auc_file.write('Micro auc is:\n')
        auc_file.write(str(rf_test_auc_micro))
        auc_file.write('\n')

        auc_file.write('Macro auc is:\n')
        auc_file.write(str(rf_test_auc_macro))
        auc_file.write('\n')

        auc_file.write('Weighted auc is:\n')
        auc_file.write(str(rf_test_auc_weighted))
        auc_file.write('\n')
        
        auc_file.write(','.join(test_data_y.columns))
        auc_file.write('\n')
        auc_file.write(','.join([str(x) for x in rf_test_auc_class_based]))



def xgboost_model(train_data_path
                        ,test_data_path
                        ,path_to_features
                        ,top_n_features
                        # ,num_target
                        ,dropped_targets):

    # num_target = num_target - len(dropped_targets)
    # with open('results/classical_ml_models/rf_model.pkl', 'rb') as f:
    #     clf2 = pickle.load(f)
    print('Reading the data:')
    print(train_data_path)
    print(test_data_path)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    train_data = train_data.sample(frac=1).reset_index(drop=True)  
    test_data = test_data.sample(frac=1).reset_index(drop=True) 

    target_col_names = [x for x in train_data.columns if 'target_proc_' in x]
    
    train_data_x = train_data[train_data.columns[~train_data.columns.isin(target_col_names + ['Patient_ID'])]]
    train_data_y = train_data[train_data.columns[train_data.columns.isin(target_col_names)]]

    test_data_x = test_data[test_data.columns[~test_data.columns.isin(target_col_names + ['Patient_ID'])]]
    test_data_y = test_data[test_data.columns[test_data.columns.isin(target_col_names)]]

    print('Finished reading data...')
    # n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # # Maximum number of levels in tree
    # max_depth = [4, 8, 16, 32, 64]# [int(x) for x in np.linspace(10, 110, num = 11)]
    # gamma = [0.001, 0.01, 0.1, 1, 10]
    # learning_rate = [0.0001, 0.001, 0.01, 0.1, 1]
    # # Create the random grid
    # hyperparameters = {'estimator__n_estimators': n_estimators
    #                , 'estimator__max_depth': max_depth
    #                , 'estimator__gamma': gamma
    #                , 'estimator__learning_rate':learning_rate
    #                }

    # print('Hyperparameters:')
    # print(hyperparameters)
    # with open('saved_classical_ml_models/xgb_hyperparameters_recommender.csv', 'w') as csv_file:  
    #     writer = csv.writer(csv_file)
    #     for key, value in hyperparameters.items():
    #        writer.writerow([key, value]) 
     
    optimum_params= pd.read_csv('saved_classical_ml_models/best_params_xgb.csv', header=None)
    optimum_params.columns = ['param','val']

    n_estimators_optimum = optimum_params[optimum_params['param']=='n_estimators']['val'].values[0]
    max_depth_optimum = optimum_params[optimum_params['param']=='max_depth']['val'].values[0]
    learning_rate_optimum = optimum_params[optimum_params['param']=='learning_rate']['val'].values[0]
    gamma_optimum = optimum_params[optimum_params['param']=='gamma']['val'].values[0]
    print(n_estimators_optimum)
    print(max_depth_optimum)
    print(learning_rate_optimum)
    print(gamma_optimum)

    clf = MultiOutputClassifier(xgb.XGBClassifier(booster='gbtree', n_estimators = int(n_estimators_optimum), max_depth=int(max_depth_optimum), learning_rate=learning_rate_optimum, gamma=gamma_optimum, verbosity=1, n_jobs=-1, objective='binary:logistic', use_label_encoder=False))
    # randomCV = RandomizedSearchCV(estimator=clf, param_distributions=hyperparameters, n_iter=10, cv=3,scoring="roc_auc")
    clf.fit(train_data_x, train_data_y)
    
    # pdb.set_trace()
    # clf = OneVsRestClassifier(xgb.XGBClassifier(booster='gbtree', verbosity=1, n_jobs=-1, objective='binary:logistic', use_label_encoder=False))
    # randomCV = RandomizedSearchCV(estimator=clf, param_distributions=hyperparameters, n_iter=10, cv=3,scoring="roc_auc")
    # randomCV.fit(train_data_x.iloc[:100,:], train_data_y.iloc[:100,:])
      
      
    train_data_x.to_csv('saved_classical_ml_models/train_predictors_recommender_final_xgb.csv', index=False)
    train_data_y.to_csv('saved_classical_ml_models/train_targets_recommender_final_xgb.csv', index=False)
    # === Save models
    with open('saved_classical_ml_models/xgb_model_recommender.pkl','wb') as f:
        pickle.dump(clf,f)
    # pdb.set_trace() 
    # (pd.DataFrame.from_dict(data=clf.best_params_, orient='index').to_csv('saved_classical_ml_models/best_params_xgb_recommender.csv', header=False))
    # best_rf_model = randomCV.best_estimator_
    best_rf_model = clf
    rf_predictions = best_rf_model.predict(test_data_x)    
    test_data_x.to_csv('saved_classical_ml_models/test_predictors_recommender_final_xgb.csv', index=False)
    test_data_y.to_csv('saved_classical_ml_models/test_targets_recommender_final_xgb.csv', index=False)

    np.savetxt('saved_classical_ml_models/predictions_xgb_recommender.csv', rf_predictions, delimiter=',')

    results_report = metrics.classification_report(y_true=test_data_y, y_pred=np.array(rf_predictions), output_dict=True, target_names=test_data_y.columns)
   
    results_report_df = pd.DataFrame(results_report).transpose()
    results_report_df.to_csv('results/classical_ml_models/results_xgb_recommender.csv')

    probabilities_test = best_rf_model.predict_proba(test_data_x)
    probabilities_test_reshaped = np.array(probabilities_test)[:,:,1].T

    rf_test_auc_micro = roc_auc_score(test_data_y, probabilities_test_reshaped, multi_class='ovr', average='micro')
    rf_test_auc_macro = roc_auc_score(test_data_y, probabilities_test_reshaped, multi_class='ovr', average='macro')
    rf_test_auc_weighted = roc_auc_score(test_data_y, probabilities_test_reshaped, multi_class='ovr', average='weighted')

    rf_test_auc_class_based = roc_auc_score(test_data_y, probabilities_test_reshaped, multi_class='ovr', average=None)

    with open('results/classical_ml_models/results_xgb_recommender_detailed_auc.csv', 'w') as auc_file:
        auc_file.write('Micro auc is:\n')
        auc_file.write(str(rf_test_auc_micro))
        auc_file.write('\n')

        auc_file.write('Macro auc is:\n')
        auc_file.write(str(rf_test_auc_macro))
        auc_file.write('\n')

        auc_file.write('Weighted auc is:\n')
        auc_file.write(str(rf_test_auc_weighted))
        auc_file.write('\n')
        
        auc_file.write(','.join(test_data_y.columns))
        auc_file.write('\n')
        auc_file.write(','.join([str(x) for x in rf_test_auc_class_based]))



def rf_feature_selection(train_data_path
                        , test_data_path
                        , num_target):
    # pdb.set_trace()
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    train_all_zero_features = []
    for i in range(1, num_target+1):
        if train_data[train_data.columns[-i]].sum()==0:
            train_all_zero_features.append(train_data.columns[-i])

    test_all_zero_features = []
    for i in range(1, num_target+1):
        if test_data[test_data.columns[-i]].sum()==0:
            test_all_zero_features.append(test_data.columns[-i])
    
    dropped_targets = train_all_zero_features + test_all_zero_features
    num_target = num_target - len(dropped_targets)

    train_data = train_data.drop(columns=dropped_targets)
    test_data = test_data.drop(columns=dropped_targets)

    # pdb.set_trace()
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']#, 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]#[4, 8, 16, 32]#, 64, 128]# [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]# [4, 8, 16, 32]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]#[4, 8, 16, 32]
    # Method of selecting samples for training each tree
    # Because the data set is too large, I allways bootsrap. If bootsrap=False, the whole data set is used to train each tree
    #bootstrap = [True, False]

    # ccp_alpha = [0, 0.00001, 0.0001, 0.001, 0.1, 1]
    
    # Create the random grid
    hyperparameters = {'n_estimators': n_estimators
                   ,'max_features': max_features
                   , 'max_depth': max_depth
                   , 'min_samples_split': min_samples_split
                   , 'min_samples_leaf': min_samples_leaf
                   #, 'bootstrap': bootstrap
                   }#,'ccp_alpha': ccp_alpha}
    print('Hyperparameters:')
    print(hyperparameters)
    with open('saved_classical_ml_models/rf_hyperparameters_recommender_forFS.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in hyperparameters.items():
           writer.writerow([key, value])
    # pdb.set_trace()  
    train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)      

    randomCV = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=-1, warm_start=True, verbose=1), param_distributions=hyperparameters, n_iter=50, cv=5,scoring="roc_auc")
    
    randomCV.fit(train_data.iloc[:,1:-num_target], train_data.iloc[:,-num_target:])
    # pdb.set_trace()
    # === Save models
    with open('saved_classical_ml_models/rf_model_recommender_forFS.pkl','wb') as f:
        pickle.dump(randomCV,f)
    
    (pd.DataFrame.from_dict(data=randomCV.best_params_, orient='index').to_csv('saved_classical_ml_models/best_params_rf_recommender_forFS.csv', header=False))
    best_rf_model= randomCV.best_estimator_
    
    feat_importances = pd.Series(randomCV.best_estimator_.feature_importances_, index=train_data.iloc[:,1:-num_target].columns)
    feat_importances.to_csv('saved_classical_ml_models/feature_impoerance_rf_recommender.csv')
    
    return 'saved_classical_ml_models/feature_impoerance_rf_recommender.csv', dropped_targets



def baseline_models(test_data_path):

    # pdb.set_trace()
    print('Reading the data:')
    print(test_data_path)
    test_data = pd.read_csv(test_data_path)
    test_data = test_data.sample(frac=1).reset_index(drop=True) 

    target_col_names = [x for x in test_data.columns if 'target_proc_' in x]
    target_col_ids = [int(x.replace('target_proc_', '')) for x in target_col_names]


    # A model that always predict top 10
    frequent_procs = pd.read_csv('intermediate_files/procedure_id_frequencies_mci.csv')
    frequent_procs.columns = frequent_procs.columns.str.strip()
    frequent_procs = frequent_procs.sort_values('num patient', ascending=False) 
    frequent_procs_selected = frequent_procs[frequent_procs['Code'].isin(target_col_ids)]
    frequent_procs_top_n_id = frequent_procs_selected['Code'].values[:10].tolist()
    frequent_procs_top_n_names = ['target_proc_'+str(x) for x in frequent_procs_top_n_id]

    # pdb.set_trace()
    with open('results/classical_ml_models/reapeated_testing/reapeated_testing_top10_recommender.csv', 'w') as res_file:   
        res_file.write('Micro avg precision, Micro avg recall, Micro avg F1, Micro avg AUC\n')
        for cv_idx in range(10):
            current_test_fold_stationary = pd.read_csv('saved_classical_ml_models/kfolds_testing_recommender/test_fold'+str(cv_idx)+'_recommender.csv')
            current_test_fold_ids = current_test_fold_stationary['Patient_ID'].values
            test_data_current_fold = test_data[test_data['Patient_ID'].isin(current_test_fold_ids)]

            test_data_x = test_data_current_fold[test_data_current_fold.columns[~test_data_current_fold.columns.isin(target_col_names + ['Patient_ID'])]]
            test_data_y = test_data_current_fold[test_data_current_fold.columns[test_data_current_fold.columns.isin(target_col_names)]]

            predictions_df = pd.DataFrame(data=np.zeros((len(test_data_y), len(target_col_names))), columns=target_col_names)
            predictions_df[frequent_procs_top_n_names] = 1
            np.savetxt('saved_classical_ml_models/kfolds_testing_recommender/predictions_top10_recommender'+str(cv_idx)+'.csv', predictions_df, delimiter=',')

            results_report = metrics.classification_report(y_true=test_data_y, y_pred=np.array(predictions_df.values), output_dict=True, target_names=test_data_y.columns)
        
            results_report_df = pd.DataFrame(results_report).transpose()
            results_report_df.to_csv('results/classical_ml_models/reapeated_testing/results_top10_recommender'+str(cv_idx)+'.csv')
        
            probabilities_df = pd.DataFrame(data=np.zeros((len(test_data_y), len(target_col_names))), columns=target_col_names)
            for i in range(len(target_col_names)):
                if target_col_names[i] in frequent_procs_top_n_names:
                    probabilities_df[target_col_names[i]] = np.random.uniform(0.5,1, size=len(test_data_y))
                else:
                    probabilities_df[target_col_names[i]] = np.random.uniform(0,0.5, size=len(test_data_y))   
            test_auc_micro = roc_auc_score(test_data_y, probabilities_df, multi_class='ovr', average='micro')
            
            res_file.write(str(results_report_df['precision'].loc['micro avg']))
            res_file.write(',')

            res_file.write(str(results_report_df['recall'].loc['micro avg']))
            res_file.write(',')

            res_file.write(str(results_report_df['f1-score'].loc['micro avg']))
            res_file.write(',')
            
            res_file.write(str(test_auc_micro))
            res_file.write('\n')
    # pdb.set_trace()

    with open('results/classical_ml_models/reapeated_testing/reapeated_testing_random_recommender.csv', 'w') as res_file:   
        res_file.write('Micro avg precision, Micro avg recall, Micro avg F1, Micro avg AUC\n')
        for cv_idx in range(10):
            current_test_fold_stationary = pd.read_csv('saved_classical_ml_models/kfolds_testing_recommender/test_fold'+str(cv_idx)+'_recommender.csv')
            current_test_fold_ids = current_test_fold_stationary['Patient_ID'].values
            test_data_current_fold = test_data[test_data['Patient_ID'].isin(current_test_fold_ids)]

            test_data_x = test_data_current_fold[test_data_current_fold.columns[~test_data_current_fold.columns.isin(target_col_names + ['Patient_ID'])]]
            test_data_y = test_data_current_fold[test_data_current_fold.columns[test_data_current_fold.columns.isin(target_col_names)]]


            predictions = np.random.randint(0,2,size=(len(test_data_y),len(target_col_names)))
            np.savetxt('saved_classical_ml_models/kfolds_testing_recommender/predictions_random_recommender'+str(cv_idx)+'.csv', predictions, delimiter=',')
            
            results_report = metrics.classification_report(y_true=test_data_y, y_pred=np.array(predictions), output_dict=True, target_names=test_data_y.columns)
            results_report_df = pd.DataFrame(results_report).transpose()
            results_report_df.to_csv('results/classical_ml_models/reapeated_testing/results_random_recommender'+str(cv_idx)+'.csv')
            
            probabilities_test_reshaped = np.random.uniform(0,1, size=(len(test_data_y),len(target_col_names)))
            test_auc_micro = roc_auc_score(test_data_y, probabilities_test_reshaped, multi_class='ovr', average='micro')

            res_file.write(str(results_report_df['precision'].loc['micro avg']))
            res_file.write(',')

            res_file.write(str(results_report_df['recall'].loc['micro avg']))
            res_file.write(',')

            res_file.write(str(results_report_df['f1-score'].loc['micro avg']))
            res_file.write(',')
            
            res_file.write(str(test_auc_micro))
            res_file.write('\n')


def repeated_test(trained_rf_path
                        , trained_xgb_path
                        , test_data_path
                        ):
    # pdb.set_trace()
    num_test_folds = 10
    test_data = pd.read_csv(test_data_path)
    target_col_names = [x for x in test_data.columns if 'target_proc_' in x]

    test_data_x = test_data[test_data.columns[~test_data.columns.isin(target_col_names + ['Patient_ID'])]]
    test_data_y = test_data[test_data.columns[test_data.columns.isin(target_col_names)]]

    randomCV = pickle.load(open(trained_rf_path, 'rb'))
    best_model = randomCV.best_estimator_

    with open('results/classical_ml_models/reapeated_testing/reapeated_testing_rf_recommender.csv', 'w') as res_file:   
        res_file.write('Micro avg precision, Micro avg recall, Micro avg F1, Micro avg AUC\n')

        test_data_kfolds = np.array_split(test_data, num_test_folds)  

        fold_counter=0 
        for i in range(len(test_data_kfolds)):
            current_test_fold = test_data_kfolds[i]

            current_test_fold_x = current_test_fold[current_test_fold.columns[~current_test_fold.columns.isin(target_col_names + ['Patient_ID'])]]
            current_test_fold_y = current_test_fold[current_test_fold.columns[current_test_fold.columns.isin(target_col_names)]]
            current_test_fold.to_csv('saved_classical_ml_models/kfolds_testing_recommender/test_fold'+str(fold_counter)+'_recommender.csv', index=False)
            fold_counter +=1
            
            predictions = best_model.predict(current_test_fold_x)    
            results_report = metrics.classification_report(y_true=current_test_fold_y, y_pred=np.array(predictions), output_dict=True, target_names=current_test_fold_y.columns)
            results_report_df = pd.DataFrame(results_report).transpose()
            results_report_df.to_csv('results/classical_ml_models/reapeated_testing/results_rf_recommender_'+str(fold_counter)+'.csv')
            
            res_file.write(str(results_report_df['precision'].loc['micro avg']))
            res_file.write(',')

            res_file.write(str(results_report_df['recall'].loc['micro avg']))
            res_file.write(',')

            res_file.write(str(results_report_df['f1-score'].loc['micro avg']))
            res_file.write(',')
            
            probabilities_test = best_model.predict_proba(current_test_fold_x)
            probabilities_test_reshaped = np.array(probabilities_test)[:,:,1].T
            test_auc_micro = roc_auc_score(current_test_fold_y, probabilities_test_reshaped, multi_class='ovr', average='micro')

            res_file.write(str(test_auc_micro))
            res_file.write('\n')


    best_model = pickle.load(open(trained_xgb_path, 'rb'))

    with open('results/classical_ml_models/reapeated_testing/reapeated_testing_xgb_recommender.csv', 'w') as res_file:   
        res_file.write('Micro avg precision, Micro avg recall, Micro avg F1, Micro avg AUC\n')

        fold_counter=0 
        for i in range(len(test_data_kfolds)):
            current_test_fold = test_data_kfolds[i]

            current_test_fold_x = current_test_fold[current_test_fold.columns[~current_test_fold.columns.isin(target_col_names + ['Patient_ID'])]]
            current_test_fold_y = current_test_fold[current_test_fold.columns[current_test_fold.columns.isin(target_col_names)]]
            fold_counter +=1
            
            predictions = best_model.predict(current_test_fold_x)    
            results_report = metrics.classification_report(y_true=current_test_fold_y, y_pred=np.array(predictions), output_dict=True, target_names=current_test_fold_y.columns)
            results_report_df = pd.DataFrame(results_report).transpose()
            results_report_df.to_csv('results/classical_ml_models/reapeated_testing/results_xgb_recommender_'+str(fold_counter)+'.csv')
            
            res_file.write(str(results_report_df['precision'].loc['micro avg']))
            res_file.write(',')

            res_file.write(str(results_report_df['recall'].loc['micro avg']))
            res_file.write(',')

            res_file.write(str(results_report_df['f1-score'].loc['micro avg']))
            res_file.write(',')
            
            probabilities_test = best_model.predict_proba(current_test_fold_x)
            probabilities_test_reshaped = np.array(probabilities_test)[:,:,1].T
            test_auc_micro = roc_auc_score(current_test_fold_y, probabilities_test_reshaped, multi_class='ovr', average='micro')

            res_file.write(str(test_auc_micro))
            res_file.write('\n')
