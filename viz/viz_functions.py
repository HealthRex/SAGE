import pdb
import matplotlib as mpl 
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from scipy.stats import mannwhitneyu
# from scipy.stats import ttest_ind
# import math
import shap
import pickle
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.ensemble import RandomForestClassifier
import csv
# import xgboost as xgb
# from sklearn.metrics import roc_auc_score
# import multiprocessing
# import lightgbm as lgb


def tSNE_visualization(train_stationary_filename 
                                , test_stationary_filename     
                                , sampled     
                                , sample_size
                                , features_to_show
                                , perplex
                                , num_it
                                , lr_rate):
    # pdb.set_trace()
    train_data=pd.read_csv(train_stationary_filename, index_col='Patient_ID')
    test_data=pd.read_csv(test_stationary_filename, index_col='Patient_ID')    
    data_all = pd.concat([train_data, test_data])

    if sampled ==1:
        data_pos = data_all[data_all['Label'] == 1]
        data_neg = data_all[data_all['Label'] == 0]
        data_pos_sampled = data_pos.sample(n=int(sample_size/2), replace=False)
        data_neg_sampled = data_neg.sample(n=int(sample_size/2), replace=False)
        data = pd.concat([data_pos_sampled, data_neg_sampled])
        data=data.sample(frac=1)
        file_name = 'sampled_' + str(sample_size)         
    else:
        data=data_all.sample(frac=1)
        file_name = 'alldata'

    data['Label'] = data['Label'].map({1: 'MCI-positive', 0: 'MCI-negative'})

    if features_to_show == "meds_diags_procs":
        data_treatment_ready=data.iloc[:,3:-1]
        features = 'meds_diags_procs'
    elif features_to_show == "all":    
        data_treatment_ready=data.iloc[:,:-1]
        features = 'allfeatures'

    tsne_model = TSNE(n_components=2, perplexity=perplex, n_iter=num_it)
    tsne_results = tsne_model.fit_transform(data_treatment_ready)
    print("kl divergence is: ",tsne_model.kl_divergence_)
    df_tsne_results = pd.DataFrame({'Patient_ID': data.index, 'First dimension of tSNE': tsne_results[:,0], 'Second dimension of tSNE': tsne_results[:,1], 'Label': data['Label']})
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('tSNE 1', fontsize = 15)
    ax.set_ylabel('tSNE 2', fontsize = 15)
    targets = ['MCI-negative', "MCI-positive"]
    colors = ['b', 'r']
    for target, color in zip(targets,colors):
        indicesToKeep = df_tsne_results['Label'] == target
        ax.scatter(df_tsne_results.loc[indicesToKeep, 'First dimension of tSNE']
                   , df_tsne_results.loc[indicesToKeep, 'Second dimension of tSNE']
                   , c = color
                   , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.savefig('results/visualization_results/tsne_'+file_name + '_' + features +'_'+str(perplex)+'_'+str(num_it)+'_'+str(lr_rate)+'_'+ str(tsne_model.kl_divergence_)[:5]+'.png', dpi=600)
    if sampled ==1:
        data.to_csv('results/visualization_results/sampled_data_forVis_'+ file_name + '_' + features +'_'+str(perplex)+'_'+str(num_it)+'_'+str(lr_rate)+'_'+ str(tsne_model.kl_divergence_)[:5]+'.csv')
    df_tsne_results.to_csv('results/visualization_results/tSNE_results_'+ file_name+ '_' + features +'_'+str(perplex)+'_'+str(num_it)+'_'+str(lr_rate)+'_'+ str(tsne_model.kl_divergence_)[:5]+'.csv')

def pca_visualization(train_stationary_filename 
                                , test_stationary_filename     
                                , sampled     
                                , sample_size
                                , features_to_show
                                ):
    pdb.set_trace()
    train_data=pd.read_csv(train_stationary_filename, index_col='Patient_ID')
    test_data=pd.read_csv(test_stationary_filename, index_col='Patient_ID')    
    data_all = pd.concat([train_data, test_data])

    if sampled ==1:
        data_pos = data_all[data_all['Label'] == 1]
        data_neg = data_all[data_all['Label'] == 0]
        data_pos_sampled = data_pos.sample(n=int(sample_size/2), replace=False)
        data_neg_sampled = data_neg.sample(n=int(sample_size/2), replace=False)
        data = pd.concat([data_pos_sampled, data_neg_sampled])
        data=data.sample(frac=1)
        file_name = 'sampled_' + str(sample_size)         
    else:
        data=data_all.sample(frac=1)
        file_name = 'alldata'

    data['Label'] = data['Label'].map({1: 'MCI-positive', 0: 'MCI-negative'})

    if features_to_show == "meds_diags_procs":
        data_treatment_ready=data.iloc[:,3:-1]
        features = 'meds_diags_procs'
    elif features_to_show == "all":    
        data_treatment_ready=data.iloc[:,:-1]
        features = 'allfeatures'
    # pdb.set_trace()
    data_treatment_std = StandardScaler().fit_transform(data_treatment_ready)
    pca_model = PCA(n_components=2)
    data_pca = pca_model.fit_transform(data_treatment_std)

    pca_c = pca_model.components_
    pca_ev = pca_model.explained_variance_
    pca_evr = pca_model.explained_variance_ratio_
    with open('results/visualization_results/pca_stat.csv', 'w') as pca_file:
        pca_file.write('components_ are:')
        pca_file.write('\n')
        for i in range(2):
            pca_file.write(','.join(map(str, pca_c[i])))
            pca_file.write('\n')
        pca_file.write('explained_variance_ are:')
        pca_file.write('\n')
        pca_file.write(','.join(map(str, pca_ev)))
        pca_file.write('\n')        
        pca_file.write('explained_variance_ratio_ are:')
        pca_file.write('\n')
        pca_file.write(','.join(map(str, pca_evr)))
        pca_file.write('\n')        
    pdb.set_trace()

    df_pca_final = pd.DataFrame({'Patient_ID': data.index, 'principal component 1': data_pca[:,0], 'principal component 2': data_pca[:,1], 'Label': data['Label']})
    df_pca_final.to_csv('results/visualization_results/Final_PCA.csv')

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    targets = ['MCI-positive', 'MCI-negative']
    colors = ['r', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = df_pca_final['Label'] == target
        ax.scatter(df_pca_final.loc[indicesToKeep, 'principal component 1']
                   , df_pca_final.loc[indicesToKeep, 'principal component 2']
                   , c = color
                   , s = 50)
                   #,alpha=0.03)
    ax.legend(targets)
    ax.grid()
    plt.savefig('results/visualization_results/pca_'+file_name + '_' + features +'.png', dpi=600)


def plot_shaps_from_saved_model (test_stationary_filename
                , trained_model_path                     
            ):
    pdb.set_trace()
    print('Reading the test data ....')
    test_data = pd.read_csv(test_stationary_filename)
    test_data = test_data.sample(frac=0.21).reset_index(drop=True) 

    randomCV =  pickle.load(open(trained_model_path, 'rb'))
    model = randomCV.best_estimator_
    explainer = shap.Explainer(model, test_data.iloc[:,:-2])
    shap_values = explainer(test_data.iloc[:,:-2])
    shap.plots.beeswarm(shap_values, plot_size=[14,8])    

    data_all = test_data_features_filtered

    print('Sampling the data ....')
    data_pos = data_all[data_all['Label'] == 1]
    data_neg = data_all[data_all['Label'] == 0]
    data_pos_sampled = data_pos.sample(n=int(sample_size_for_shap * len(data_pos)), replace=False)
    data_neg_sampled = data_neg.sample(n=int(sample_size_for_shap * len(data_neg)), replace=False)
    data_all_sampled = pd.concat([data_pos_sampled, data_neg_sampled])
    data_all_sampled=data_all_sampled.sample(frac=1)

    data_all_sampled.to_csv('results/visualization_results/data_all_sampled_using_trained_model.csv', index=False)

    print('===========================')
    print('Loading XGB model ....')
    # trained_model_path = 'results/visualization_results/shap_results_sep_12/xgb_model.pkl'
    randomCV =  pickle.load(open(trained_model_path, 'rb'))
    model = randomCV.best_estimator_

    print('===========================')
    print('Creating shap plots ....')    
    explainer = shap.Explainer(model, data_all_sampled.iloc[:,1:-1])
    shap_values = explainer(data_all_sampled.iloc[:,1:-1])
    shap.plots.beeswarm(shap_values, plot_size=[14,8])    
    # fig = plt.gcf()
    # fig.set_size_inches(18, 12)
    plt.savefig('results/visualization_results/beeswarm_xgb_using_trained_model.png', dpi=600)
    plt.close()


    predictions = model.predict(data_all_sampled.iloc[:,1:-1])

    np.savetxt('results/visualization_results/predictions_xgb_for_shap_using_trained_model.csv', predictions, delimiter=',')

    tn, tp, fn, fp, accuracy, precision, recall, specificity, F1, rf_test_auc = performance_evaluation(predictions
                                                                            , data_all_sampled
                                                                            , model
                                                                            )   
    write_results(tn, tp, fn, fp, 
                accuracy, precision, recall, specificity
                , F1, rf_test_auc
                , 'trained_xgb')
    # pdb.set_trace()
    print('Test')


def plot_prevalance(train_stationary_filename 
                                , test_stationary_filename     
                                , features_to_show
                                , mci_metadata_path
                                , nonmci_metadata_path
                                ):
    pdb.set_trace()

    print('Reading training and testing data from {} and {}'.format(train_stationary_filename, test_stationary_filename))
    train_data=pd.read_csv(train_stationary_filename)
    test_data=pd.read_csv(test_stationary_filename)    
    data_all = pd.concat([train_data, test_data])
    data_all_pos = data_all[data_all['Label']==1]
    data_all_neg = data_all[data_all['Label']==0]

    train_data[train_data['Patient_ID'] == 'JC2a00006']
    test_data[test_data['Patient_ID'] == 'JC2a00006']

    mci_metadata = pd.read_csv(mci_metadata_path)
    nonmci_metadata = pd.read_csv(nonmci_metadata_path)

    mci_metadata_cohort = mci_metadata[mci_metadata['anon_id'].isin(data_all_pos['Patient_ID'].values.tolist())]
    nonmci_metadata_cohort = nonmci_metadata[nonmci_metadata['anon_id'].isin(data_all_neg['Patient_ID'].values.tolist())]
