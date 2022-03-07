# from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import pdb
# from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
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
import shap
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier


def performance_evaluation(rf_predictions
                        , test_data_for_eval
                        , best_model):
    labels = test_data_for_eval['Label'].values
    rf_test_auc=roc_auc_score(test_data_for_eval['Label'], best_model.predict_proba(test_data_for_eval.drop(['Patient_ID', 'Label'], axis=1, inplace=False))[:,1])
    tp=0
    tn=0
    fn=0
    fp=0
    accuracy=0
    precision=0
    recall=0
    F1=0
    specificity=0
    for asses_ind in range(len(rf_predictions)):
        if(rf_predictions[asses_ind]==0 and labels[asses_ind]==0):
            tn=tn+1
        elif(rf_predictions[asses_ind]==0 and labels[asses_ind]==1):
            fn=fn+1
        elif(rf_predictions[asses_ind]==1 and labels[asses_ind]==1):
            tp=tp+1
        elif(rf_predictions[asses_ind]==1 and labels[asses_ind]==0):    
            fp=fp+1
    accuracy=(tn+tp)/(tn+tp+fn+fp)
    if(tp+fp == 0):
        precision=0
    else:
        precision=tp/(tp+fp)
    if(tp+fn==0):
        recall=0
    else:
        recall=tp/(tp+fn)
    if(precision==0 and recall==0):
        F1=0
    else:            
        F1=(2*precision*recall)/(precision+recall)
    if(tn+fp==0):
        specificity= 0
    else:
        specificity= tn/(tn+fp)    

    return tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, rf_test_auc    

def write_results(tn
                , tp
                , fn
                , fp
                , accuracy
                , precision
                , recall
                , specificity
                , F1
                , rf_test_auc
                , model
                    ):
    with open('results/classical_ml_models/'+model+'_prediction_performance.csv', 'w') as f_results:
        f_results.write("Precision is: ")
        f_results.write(str(precision))
        f_results.write("\n")
        
        f_results.write("Recall is: ")
        f_results.write(str(recall))
        f_results.write("\n")
        
        f_results.write("Accuracy is: ")
        f_results.write(str(accuracy))
        f_results.write("\n") 

        f_results.write("F1 is: ")
        f_results.write(str(F1))
        f_results.write("\n")

        f_results.write("Specificity is: ")
        f_results.write(str(specificity))
        f_results.write("\n")

        f_results.write("AUC is: ")
        f_results.write(str(rf_test_auc))
        f_results.write("\n")

        f_results.write("TP is: ")
        f_results.write(str(tp))
        f_results.write("\n")

        f_results.write("TN is: ")
        f_results.write(str(tn))
        f_results.write("\n")

        f_results.write("FP is: ")
        f_results.write(str(fp))
        f_results.write("\n")

        f_results.write("FN is: ")
        f_results.write(str(fn))
        f_results.write("\n")



# === XGBoost
def xgboost_model(train_data_path
                        ,test_data_path
                        ,path_to_features
                        ,top_n_features):

    # pdb.set_trace()
    print('Reading the data:')
    print(train_data_path)
    print(test_data_path)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    feature_ranking = pd.read_csv(path_to_features, names=['Feature', 'Score']).sort_values(by='Score', ascending=False)
    selected_features = feature_ranking.iloc[:top_n_features, 0].values.tolist()
    selected_features = selected_features + ['Label', 'Patient_ID']
    train_data = train_data[selected_features]
    test_data = test_data[selected_features]


    print('Finished reading data...')
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Maximum number of levels in tree
    max_depth = [4, 8, 16, 32, 64, 128]# [int(x) for x in np.linspace(10, 110, num = 11)]
    gamma = [0.001, 0.01, 0.1, 1, 10]
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 1]
    # Create the random grid
    hyperparameters = {'n_estimators': n_estimators
                   , 'max_depth': max_depth
                   , 'gamma': gamma
                   , 'learning_rate':learning_rate
                   }

    print('Hyperparameters:')
    print(hyperparameters)
    with open('saved_classical_ml_models/xgb_hyperparameters.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in hyperparameters.items():
           writer.writerow([key, value])
    train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)  
    test_data_shuffled = test_data.sample(frac=1).reset_index(drop=True)  

    # pdb.set_trace()
    randomCV = RandomizedSearchCV(estimator=xgb.XGBClassifier(booster='gbtree', verbosity=1, n_jobs=-1, objective='binary:logistic', use_label_encoder=False), param_distributions=hyperparameters, n_iter=50, cv=3,scoring="roc_auc")
    randomCV.fit(train_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False), train_data_shuffled['Label'])
    
    # === Save models
    with open('saved_classical_ml_models/xgb_model.pkl','wb') as f:
        pickle.dump(randomCV,f)
    
    (pd.DataFrame.from_dict(data=randomCV.best_params_, orient='index').to_csv('saved_classical_ml_models/best_params_xgb.csv', header=False))
    best_xgb_model= randomCV.best_estimator_

    xgb_predictions = best_xgb_model.predict(test_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False))    
    np.savetxt('saved_classical_ml_models/predictions_xgb.csv', xgb_predictions, delimiter=',')

    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, xgb_test_auc = performance_evaluation(xgb_predictions
                                                                            , test_data_shuffled
                                                                            , best_xgb_model
                                                                            )   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, xgb_test_auc
                , 'xgb')
    # pdb.set_trace()
    metrics.plot_roc_curve(best_xgb_model, test_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False), test_data_shuffled['Label'], name='Random Forest') 
    plt.savefig('results/classical_ml_models/roc_curve_xgb.png', dpi=300)
    plt.close()

    # pdb.set_trace()
    print('Creating shap plots using test data ....')    

    explainer = shap.TreeExplainer(best_xgb_model)
    shap_values = explainer(test_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False))
    
    # pdb.set_trace()
    # f = plt.figure()
    # f.set_size_inches(18, 12)
    # shap.plots.scatter(shap_values[:,"age"], color=shap_values)
    # f.savefig('results/visualization_results/scatter_Age_xgb_original_testdata.png', dpi=600)

    f = plt.figure()
    shap.plots.beeswarm(shap_values)
    f.savefig('results/visualization_results/beeswarm_xgb_original_test.png', dpi=600)

    f2 = plt.figure()
    shap.plots.bar(shap_values)
    f2.savefig('results/visualization_results/bar_xgb_original_test.png', dpi=600)

    plt.close()
    shap.plots.beeswarm(shap_values)    
    fig = plt.gcf()
    fig.set_size_inches(18, 12)
    fig.savefig('results/visualization_results/beeswarm_xgb_resized_test.png', dpi=600)
    plt.close(fig)

    with open('results/visualization_results/shap_values_uing_xgb_and_tes_data.pkl','wb') as f:
        pickle.dump(shap_values,f)

    # # pdb.set_trace()
    # print('Creating shap plots using all data ....')    
    # data_all_for_shap = train_data_shuffled.append([test_data_shuffled], ignore_index=True)
    # f = plt.figure()
    # shap.plots.scatter(shap_values[:,"age"], color=shap_values)
    # f.savefig('results/visualization_results/scatter_Age_xgb_original_alldata.png', dpi=600)

    # f = plt.figure()
    # shap.plots.beeswarm(shap_values)
    # f.savefig('results/visualization_results/beeswarm_xgb_original_alldata.png', dpi=600)

    # f = plt.figure()
    # shap.plots.bar(shap_values)
    # f.savefig('results/visualization_results/bar_xgb_original_alldata.png', dpi=600)
    

# === Logistic regression
def logistic_regression_model(train_data_path
                        ,test_data_path
                        ,path_to_features
                        ,top_n_features):

    # pdb.set_trace()
    print('Reading the data:')
    print(train_data_path)
    print(test_data_path)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    feature_ranking = pd.read_csv(path_to_features, names=['Feature', 'Score']).sort_values(by='Score', ascending=False)
    selected_features = feature_ranking.iloc[:top_n_features, 0].values.tolist()
    selected_features = selected_features + ['Label', 'Patient_ID']
    train_data = train_data[selected_features]
    test_data = test_data[selected_features]

    print('Finished reading data...')
    hyperparameters = {
                        'penalty' : ['l2', 'none'],
                        'solver' : ['sag','saga'],
                        # 'degree' : [0, 1, 2, 3, 4, 5, 6],
                        #'penalty': ['l1', 'l2'],
                        #'loss' : ['hinge', 'squared_hinge'],
                        #'dual' : [True, False],
                        'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]#[2**-10, 2** -8, 2 ** -6, 2** -4, 2**-2, 1, 2**2, 2**4, 2**6, 2**8, 2**10]       
        }
    print('Hyperparameters:')
    print(hyperparameters)
    with open('saved_classical_ml_models/lr_hyperparameters.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in hyperparameters.items():
           writer.writerow([key, value])
    
    train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)  
    test_data_shuffled = test_data.sample(frac=1).reset_index(drop=True)  
    # pdb.set_trace()
    randomCV = RandomizedSearchCV(estimator=LogisticRegression(n_jobs=-1, verbose=1), param_distributions=hyperparameters, n_iter=50, cv=3,scoring="roc_auc")
    randomCV.fit(train_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False), train_data_shuffled['Label'])


    # === Save models
    with open('saved_classical_ml_models/lr_model.pkl','wb') as f:
        pickle.dump(randomCV,f)
    
    (pd.DataFrame.from_dict(data=randomCV.best_params_, orient='index').to_csv('saved_classical_ml_models/best_params_lr.csv', header=False))
    best_lr_model= randomCV.best_estimator_

    lr_predictions = best_lr_model.predict(test_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False))    
    np.savetxt('saved_classical_ml_models/predictions_lr.csv', lr_predictions, delimiter=',')

    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, lr_test_auc = performance_evaluation(lr_predictions
                                                                            , test_data_shuffled
                                                                            , best_lr_model
                                                                            )   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, lr_test_auc
                , 'lr')
    # pdb.set_trace()
    metrics.plot_roc_curve(best_lr_model, test_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False), test_data_shuffled['Label'], name='Random Forest') 
    plt.savefig('results/classical_ml_models/roc_curve_lr.png', dpi=300)
    plt.close()





def random_forest_model(train_data_path
                        ,test_data_path
                        ,path_to_features
                        ,top_n_features):


    print('Reading the data:')
    print(train_data_path)
    print(test_data_path)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    
    feature_ranking = pd.read_csv(path_to_features, names=['Feature', 'Score']).sort_values(by='Score', ascending=False)
    selected_features = feature_ranking.iloc[:top_n_features, 0].values.tolist()
    selected_features = selected_features + ['Label', 'Patient_ID']
    train_data = train_data[selected_features]
    test_data = test_data[selected_features]

    
    ax= feature_ranking.iloc[:top_n_features,:].plot.bar(x='Feature',y='Score')
    ax.get_figure().savefig("results/visualization_results/feature_importance_rf.png", dpi=300)



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
    print('Hyperparameters:')
    print(hyperparameters)
    with open('saved_classical_ml_models/rf_hyperparameters.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in hyperparameters.items():
           writer.writerow([key, value])
    train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)  
    test_data_shuffled = test_data.sample(frac=1).reset_index(drop=True)  

    randomCV = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=-1, warm_start=True, verbose=1), param_distributions=hyperparameters, n_iter=50, cv=5,scoring="roc_auc")
    randomCV.fit(train_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False), train_data_shuffled['Label'])
    # pdb.set_trace()
    # === Save models
    with open('saved_classical_ml_models/rf_model.pkl','wb') as f:
        pickle.dump(randomCV,f)
    
    (pd.DataFrame.from_dict(data=randomCV.best_params_, orient='index').to_csv('saved_classical_ml_models/best_params_rf.csv', header=False))
    best_rf_model= randomCV.best_estimator_

    rf_predictions = best_rf_model.predict(test_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False))    
    np.savetxt('saved_classical_ml_models/predictions_rf.csv', rf_predictions, delimiter=',')

    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, rf_test_auc = performance_evaluation(rf_predictions
                                                                            , test_data_shuffled
                                                                            , best_rf_model
                                                                            )   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, rf_test_auc
                , 'rf')
    # pdb.set_trace()
    metrics.plot_roc_curve(best_rf_model, test_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False), test_data_shuffled['Label'], name='Random Forest') 
    plt.savefig('results/classical_ml_models/roc_curve_rf.png', dpi=300)
    plt.close()


def rf_feature_selection(train_data_path):
    train_data = pd.read_csv(train_data_path)
    
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
    with open('saved_classical_ml_models/rf_hyperparameters_forFS.csv', 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in hyperparameters.items():
           writer.writerow([key, value])
    train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)      

    randomCV = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=-1, warm_start=True, verbose=1), param_distributions=hyperparameters, n_iter=50, cv=5,scoring="roc_auc")
    randomCV.fit(train_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False), train_data_shuffled['Label'])
    # pdb.set_trace()
    # === Save models
    with open('saved_classical_ml_models/rf_model_forFS.pkl','wb') as f:
        pickle.dump(randomCV,f)
    
    (pd.DataFrame.from_dict(data=randomCV.best_params_, orient='index').to_csv('saved_classical_ml_models/best_params_rf_forFS.csv', header=False))
    best_rf_model= randomCV.best_estimator_
    
    feat_importances = pd.Series(randomCV.best_estimator_.feature_importances_, index=train_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False).columns)
    feat_importances.to_csv('saved_classical_ml_models/feature_impoerance_rf.csv')
    



def test_with_imb(trained_rf_path
                , trained_lr_path
                , trained_xgb_path
                , train_data_path
                , test_data_path
                # , stationary_mci_path
                , stationary_nonmci_path
                , path_to_features
                , top_n_features                
                        ):
    # pdb.set_trace()
    print('Testing using imbalance test sets. ')
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    stationary_nonmci_data = pd.read_csv(stationary_nonmci_path)

    excluding_patients = test_data['Patient_ID'].values.tolist() + train_data['Patient_ID'].values.tolist()

    stationary_nonmci_data_new = stationary_nonmci_data[~stationary_nonmci_data['Patient_ID'].isin(excluding_patients)]
    stationary_nonmci_data_new.columns = stationary_nonmci_data_new.columns.str.strip()

    min_max_data = pd.read_csv('stationary_data/stationary_data_imbratio1.csv')        
    min_max_data.columns = min_max_data.columns.str.strip()

    if sum(min_max_data.columns != stationary_nonmci_data_new.columns) > 0:
        pdb.set_trace()
        print('Warning')

    epsil = 2.220446049250313e-16
    round_precision = 5
    mins = min_max_data.iloc[:,1:].min()
    maxes = min_max_data.iloc[:,1:].max()

    stationary_nonmci_data_new_normalized = (stationary_nonmci_data_new.iloc[:,1:] -mins)/((maxes-mins) + epsil)
    stationary_nonmci_data_new_normalized['Patient_ID'] = stationary_nonmci_data_new['Patient_ID']
    stationary_nonmci_data_new_normalized['Label'] = stationary_nonmci_data_new['Label']
    stationary_nonmci_data_new_normalized = stationary_nonmci_data_new_normalized.round(round_precision)


    feature_ranking = pd.read_csv(path_to_features, names=['Feature', 'Score']).sort_values(by='Score', ascending=False)
    selected_features = feature_ranking.iloc[:top_n_features, 0].values.tolist()
    selected_features = selected_features + ['Label', 'Patient_ID']

    stationary_nonmci_data_new_normalized = stationary_nonmci_data_new_normalized[selected_features]

    test_data = test_data[selected_features]


    test_data_imb = pd.concat([test_data, stationary_nonmci_data_new_normalized])
    test_data_imb = test_data_imb.sample(frac=1).reset_index(drop=True)  
    test_data_imb.to_csv('stationary_data/test_data_imb.csv',  index=False)
    # RF model 
    randomCV = pickle.load(open(trained_rf_path, 'rb'))
    best_model= randomCV.best_estimator_
    if sum(best_model.feature_names_in_ != test_data_imb.drop(['Patient_ID', 'Label'], axis=1, inplace=False).columns)>0:
        pdb.set_trace()
        print('Warning')    
    predictions = best_model.predict(test_data_imb.drop(['Patient_ID', 'Label'], axis=1, inplace=False))    
    np.savetxt('saved_classical_ml_models/imb_predictions_rf.csv', predictions, delimiter=',')
    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, test_auc = performance_evaluation(predictions, test_data_imb, best_model)   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, test_auc
                , 'rf_imb')
    metrics.plot_roc_curve(best_model, test_data_imb.drop(['Patient_ID', 'Label'], axis=1, inplace=False), test_data_imb['Label'], name='Random Forest') 
    plt.savefig('results/classical_ml_models/imb_roc_curve_rf.png', dpi=300)
    plt.close()


    randomCV = pickle.load(open(trained_lr_path, 'rb'))
    best_model= randomCV.best_estimator_
    if sum(best_model.feature_names_in_ != test_data_imb.drop(['Patient_ID', 'Label'], axis=1, inplace=False).columns)>0:
        pdb.set_trace()
        print('Warning')
    predictions = best_model.predict(test_data_imb.drop(['Patient_ID', 'Label'], axis=1, inplace=False))    
    np.savetxt('saved_classical_ml_models/imb_predictions_lr.csv', predictions, delimiter=',')
    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, test_auc = performance_evaluation(predictions, test_data_imb, best_model)   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, test_auc
                , 'lr_imb')
    metrics.plot_roc_curve(best_model, test_data_imb.drop(['Patient_ID', 'Label'], axis=1, inplace=False), test_data_imb['Label'], name='Logistic Regression') 
    plt.savefig('results/classical_ml_models/imb_roc_curve_lr.png', dpi=300)
    plt.close()

    
    randomCV = pickle.load(open(trained_xgb_path, 'rb'))
    best_model= randomCV.best_estimator_
    if sum(best_model.get_booster().feature_names != test_data_imb.drop(['Patient_ID', 'Label'], axis=1, inplace=False).columns)>0:
        pdb.set_trace()
        print('Warning')    
    predictions = best_model.predict(test_data_imb.drop(['Patient_ID', 'Label'], axis=1, inplace=False))    
    np.savetxt('saved_classical_ml_models/imb_predictions_xgb.csv', predictions, delimiter=',')
    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, test_auc = performance_evaluation(predictions, test_data_imb, best_model)   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, test_auc
                , 'xgb_imb')
    metrics.plot_roc_curve(best_model, test_data_imb.drop(['Patient_ID', 'Label'], axis=1, inplace=False), test_data_imb['Label'], name='XGBoost') 
    plt.savefig('results/classical_ml_models/imb_roc_curve_xgb.png', dpi=300)
    plt.close()
