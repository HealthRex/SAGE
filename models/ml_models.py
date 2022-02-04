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
# import shap
# import xgboost as xgb

def read_stationary_data(train_data_path
                        , validation_data_path
                        , test_data_path):
    train_data_stationary = pd.read_csv(train_data_path)
    validation_data_stationary = pd.read_csv(validation_data_path)
    test_data_stationary = pd.read_csv(test_data_path)

    return train_data_stationary, validation_data_stationary, test_data_stationary

def performance_evaluation(rf_predictions
                        , test_data_for_eval
                        , best_model):

    pdb.set_trace()
    labels = test_data_for_eval['Label'].values
    rf_test_auc=roc_auc_score(test_data_for_eval['Label'], best_model.predict_proba(test_data_for_eval.iloc[:,1:-1])[:,1])
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
    # pdb.set_trace()
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
                        ,test_data_path):

    pdb.set_trace()
    # with open('results/classical_ml_models/rf_model.pkl', 'rb') as f:
    #     clf2 = pickle.load(f)
    print('Reading the data:')
    print(train_data_path)
    print(test_data_path)
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
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
    pdb.set_trace()  
    train_data_shuffled = train_data.sample(frac=1).reset_index(drop=True)  
    test_data_shuffled = test_data.sample(frac=1).reset_index(drop=True)  
    # training_all_shuffled = training_all.sample(frac=1)
    # test_data_shuffled = test_data.sample(frac=1)
    # training_all_shuffled = shuffle(training_all, random_state=123)
    # test_data_shuffled = shuffle(test_data, random_state=123)
    
    # saving patinets IDs
    # pdb.set_trace()
    

    randomCV = RandomizedSearchCV(estimator=RandomForestClassifier(n_jobs=-1, warm_start=True, verbose=1), param_distributions=hyperparameters, n_iter=2, cv=3,scoring="roc_auc")
    randomCV.fit(train_data.drop(['Patient_ID', 'Label'], axis=1, inplace=False), train_data['Label'])
    # pdb.set_trace()
    # === Save models
    with open('saved_classical_ml_models/rf_model.pkl','wb') as f:
        pickle.dump(randomCV,f)
    
    (pd.DataFrame.from_dict(data=randomCV.best_params_, orient='index').to_csv('saved_classical_ml_models/best_params_rf.csv', header=False))
    best_rf_model= randomCV.best_estimator_
    # feat_importances = pd.Series(randomCV.best_estimator_.feature_importances_, index=training_all.iloc[:,1:-1].columns)
    # pdb.set_trace() 
    # fig = feat_importances.nlargest(len(training_all.columns)-2).plot(kind='barh', grid=True,figsize=(12,10))
    # fig.set_xlabel("Importance Score")
    # fig.set_ylabel("Features")
    # fig.get_figure().savefig("results/classical_ml_models/feature_importance.png", dpi=300)

    rf_predictions = best_rf_model.predict(test_data.drop(['Patient_ID', 'Label'], axis=1, inplace=False))    
    np.savetxt('saved_classical_ml_models/predictions_rf.csv', rf_predictions, delimiter=',')

    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, rf_test_auc = performance_evaluation(rf_predictions
                                                                            , test_data
                                                                            , best_rf_model
                                                                            )   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, rf_test_auc
                , 'rf')
    # print('Computing shap values....')
    # explainer = shap.Explainer(best_rf_model)
    # shap_values = explainer(training_all_shuffled.iloc[:,1:-1])
    # print('Finished computing shap values....')
    # pdb.set_trace()
    # print('Plotting waterfall')
    # shap.plots.waterfall(shap_values[0])
    # plt.savefig('results/classical_ml_models/waterfall_rf.png', dpi=600)
    # plt.close()    

    # print('Plotting beeswarm')
    # shap.plots.beeswarm(shap_values)
    # plt.savefig('results/classical_ml_models/beeswarm_rf.png', dpi=600)
    # plt.close()

    # print('Saving shap values in a pickle')
    # with open('results/classical_ml_models/shap_values_rf.pkl','wb') as f:
    #     pickle.dump(shap_values,f)    

    # print('Plotting abs of shap values in a bar diagram')
    # shap.plots.bar(shap_values)
    
    # pdb.set_trace()
    # print('End')

