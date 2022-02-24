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
# import xgboost as xgb


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

    randomCV = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=-1, warm_start=True, verbose=1), param_distributions=hyperparameters, n_iter=2, cv=5,scoring="roc_auc")
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

    randomCV = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=-1, warm_start=True, verbose=1), param_distributions=hyperparameters, n_iter=2, cv=5,scoring="roc_auc")
    randomCV.fit(train_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False), train_data_shuffled['Label'])
    # pdb.set_trace()
    # === Save models
    with open('saved_classical_ml_models/rf_model_forFS.pkl','wb') as f:
        pickle.dump(randomCV,f)
    
    (pd.DataFrame.from_dict(data=randomCV.best_params_, orient='index').to_csv('saved_classical_ml_models/best_params_rf_forFS.csv', header=False))
    best_rf_model= randomCV.best_estimator_
    
    feat_importances = pd.Series(randomCV.best_estimator_.feature_importances_, index=train_data_shuffled.drop(['Patient_ID', 'Label'], axis=1, inplace=False).columns)
    feat_importances.to_csv('saved_classical_ml_models/feature_impoerance_rf.csv')
    
    return 'saved_classical_ml_models/feature_impoerance_rf.csv'









