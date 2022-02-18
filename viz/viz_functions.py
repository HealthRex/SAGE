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


def compute_table_stats(train_stationary_filename 
                                , test_stationary_filename     
                                , features_to_show
                                , mci_metadata_path
                                , nonmci_metadata_path
                                , feature_ranking_path
                                ):
    pdb.set_trace()

    print('Reading training and testing data from {} and {}'.format(train_stationary_filename, test_stationary_filename))
    train_data=pd.read_csv(train_stationary_filename)
    test_data=pd.read_csv(test_stationary_filename)    
    data_all = pd.concat([train_data, test_data])
    data_all_pos = data_all[data_all['Label']==1]
    data_all_neg = data_all[data_all['Label']==0]

    mci_metadata = pd.read_csv(mci_metadata_path)
    nonmci_metadata = pd.read_csv(nonmci_metadata_path)

    mci_metadata_cohort = mci_metadata[mci_metadata['anon_id'].isin(data_all_pos['Patient_ID'].values.tolist())]
    nonmci_metadata_cohort = nonmci_metadata[nonmci_metadata['anon_id'].isin(data_all_neg['Patient_ID'].values.tolist())]

    with open('results/visualization_results/stats.csv','w') as stat_file:
        avg_age_pos = (mci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - mci_metadata_cohort['bdate'].str[:4].astype(int)).mean()
        std_age_pos = (mci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - mci_metadata_cohort['bdate'].str[:4].astype(int)).std()
        q25_age_pos = (mci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - mci_metadata_cohort['bdate'].str[:4].astype(int)).quantile(q=0.25)
        q50_age_pos = (mci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - mci_metadata_cohort['bdate'].str[:4].astype(int)).quantile(q=0.50)
        q75_age_pos = (mci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - mci_metadata_cohort['bdate'].str[:4].astype(int)).quantile(q=0.75)
        stat_file.write('Average age in cases:\n')
        stat_file.write(str(avg_age_pos))
        stat_file.write('\n')
        stat_file.write('std age in cases:\n')
        stat_file.write(str(std_age_pos))
        stat_file.write('\n')
        stat_file.write('25Perc age in cases:\n')
        stat_file.write(str(q25_age_pos))
        stat_file.write('\n')
        stat_file.write('50Perc age in cases:\n')
        stat_file.write(str(q50_age_pos))
        stat_file.write('\n')
        stat_file.write('75Perc age in cases:\n')
        stat_file.write(str(q75_age_pos))
        stat_file.write('\n')
                                        
        avg_age_neg = (nonmci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - nonmci_metadata_cohort['bdate'].str[:4].astype(int)).mean()
        std_age_neg = (nonmci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - nonmci_metadata_cohort['bdate'].str[:4].astype(int)).std()
        q25_age_neg = (nonmci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - nonmci_metadata_cohort['bdate'].str[:4].astype(int)).quantile(q=0.25)
        q50_age_neg = (nonmci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - nonmci_metadata_cohort['bdate'].str[:4].astype(int)).quantile(q=0.50)
        q75_age_neg = (nonmci_metadata_cohort['index_date_OR_diag_date'].str[:4].astype(int) - nonmci_metadata_cohort['bdate'].str[:4].astype(int)).quantile(q=0.75)
        stat_file.write('Average age in controls:\n')
        stat_file.write(str(avg_age_neg))
        stat_file.write('\n')
        stat_file.write('std age in controls:\n')
        stat_file.write(str(std_age_neg))
        stat_file.write('\n')
        stat_file.write('25Perc age in controls:\n')
        stat_file.write(str(q25_age_neg))
        stat_file.write('\n')
        stat_file.write('50Perc age in controls:\n')
        stat_file.write(str(q50_age_neg))
        stat_file.write('\n')
        stat_file.write('75Perc age in controls:\n')
        stat_file.write(str(q75_age_neg))
        stat_file.write('\n')


        # sex
        mci_metadata_cohort['sex'] = mci_metadata_cohort['sex'].str.strip()
        num_female_mci = mci_metadata_cohort[mci_metadata_cohort['sex'] == 'Female'].shape[0]
        perc_female_mci =  num_female_mci/mci_metadata_cohort.shape[0]   
        stat_file.write('number of females in cases:\n')
        stat_file.write(str(num_female_mci))
        stat_file.write('\n')
        stat_file.write('percentage of females in casess:\n')
        stat_file.write(str(perc_female_mci))
        stat_file.write('\n')

        nonmci_metadata_cohort['sex'] = nonmci_metadata_cohort['sex'].str.strip()
        num_female_nonmci = nonmci_metadata_cohort[nonmci_metadata_cohort['sex'] == 'Female'].shape[0]
        perc_female_nonmci =  num_female_nonmci/nonmci_metadata_cohort.shape[0]   
        stat_file.write('number of females in controls:\n')
        stat_file.write(str(num_female_nonmci))
        stat_file.write('\n')
        stat_file.write('percentage of females in controls:\n')
        stat_file.write(str(perc_female_nonmci))
        stat_file.write('\n')

        # Race ["White", "Other", "Asian", "Black", "Unknown", "Pacific Islander", "Native American"]
        mci_metadata_cohort['canonical_race'] = mci_metadata_cohort['canonical_race'].str.strip()
        

        num_white_mci = mci_metadata_cohort[mci_metadata_cohort['canonical_race'] == 'White'].shape[0]
        perc_white_mci =  num_white_mci/mci_metadata_cohort.shape[0]   
        stat_file.write('number of white in cases:\n')
        stat_file.write(str(num_white_mci))
        stat_file.write('\n')
        stat_file.write('percentage of white in cases:\n')
        stat_file.write(str(perc_white_mci))
        stat_file.write('\n')

        num_Other_mci = mci_metadata_cohort[mci_metadata_cohort['canonical_race'] == 'Other'].shape[0]
        perc_Other_mci =  num_Other_mci/mci_metadata_cohort.shape[0]   
        stat_file.write('number of Other race in cases:\n')
        stat_file.write(str(num_Other_mci))
        stat_file.write('\n')
        stat_file.write('percentage of Other race in cases:\n')
        stat_file.write(str(perc_Other_mci))
        stat_file.write('\n')

        num_Asian_mci = mci_metadata_cohort[mci_metadata_cohort['canonical_race'] == 'Asian'].shape[0]
        perc_Asian_mci =  num_Asian_mci/mci_metadata_cohort.shape[0]   
        stat_file.write('number of Asian in cases:\n')
        stat_file.write(str(num_Asian_mci))
        stat_file.write('\n')
        stat_file.write('percentage of Asian in cases:\n')
        stat_file.write(str(perc_Asian_mci))
        stat_file.write('\n')

        num_Black_mci = mci_metadata_cohort[mci_metadata_cohort['canonical_race'] == 'Black'].shape[0]
        perc_Black_mci =  num_Black_mci/mci_metadata_cohort.shape[0]   
        stat_file.write('number of Black in cases:\n')
        stat_file.write(str(num_Black_mci))
        stat_file.write('\n')
        stat_file.write('percentage of Black in cases:\n')
        stat_file.write(str(perc_Black_mci))
        stat_file.write('\n')

        num_Unknown_mci = mci_metadata_cohort[mci_metadata_cohort['canonical_race'] == 'Unknown'].shape[0]
        perc_Unknown_mci =  num_Unknown_mci/mci_metadata_cohort.shape[0]   
        stat_file.write('number of Unknown in cases:\n')
        stat_file.write(str(num_Unknown_mci))
        stat_file.write('\n')
        stat_file.write('percentage of Unknown in cases:\n')
        stat_file.write(str(perc_Unknown_mci))
        stat_file.write('\n')

        num_Pacific_Islander_mci = mci_metadata_cohort[mci_metadata_cohort['canonical_race'] == 'Pacific Islander'].shape[0]
        perc_Pacific_Islander_mci =  num_Pacific_Islander_mci/mci_metadata_cohort.shape[0]   
        stat_file.write('number of Pacific Islander in cases:\n')
        stat_file.write(str(num_Pacific_Islander_mci))
        stat_file.write('\n')
        stat_file.write('percentage of Pacific Islander in cases:\n')
        stat_file.write(str(perc_Pacific_Islander_mci))
        stat_file.write('\n')

        num_Native_American_mci = mci_metadata_cohort[mci_metadata_cohort['canonical_race'] == 'Native American'].shape[0]
        perc_Native_American_mci =  num_Native_American_mci/mci_metadata_cohort.shape[0]   
        stat_file.write('number of Native American in cases:\n')
        stat_file.write(str(num_Native_American_mci))
        stat_file.write('\n')
        stat_file.write('percentage of Native American in cases:\n')
        stat_file.write(str(perc_Native_American_mci))
        stat_file.write('\n')

        num_white_nonmci = nonmci_metadata_cohort[nonmci_metadata_cohort['canonical_race'] == 'White'].shape[0]
        perc_white_nonmci =  num_white_nonmci/nonmci_metadata_cohort.shape[0]   
        stat_file.write('number of white in controls:\n')
        stat_file.write(str(num_white_nonmci))
        stat_file.write('\n')
        stat_file.write('percentage of white in controls:\n')
        stat_file.write(str(perc_white_nonmci))
        stat_file.write('\n')


        num_Other_nonmci = nonmci_metadata_cohort[nonmci_metadata_cohort['canonical_race'] == 'Other'].shape[0]
        perc_Other_nonmci =  num_Other_nonmci/nonmci_metadata_cohort.shape[0]   
        stat_file.write('number of Other race in controls:\n')
        stat_file.write(str(num_Other_nonmci))
        stat_file.write('\n')
        stat_file.write('percentage of Other race in controls:\n')
        stat_file.write(str(perc_Other_nonmci))
        stat_file.write('\n')

        num_Asian_nonmci = nonmci_metadata_cohort[nonmci_metadata_cohort['canonical_race'] == 'Asian'].shape[0]
        perc_Asian_nonmci =  num_Asian_nonmci/nonmci_metadata_cohort.shape[0]   
        stat_file.write('number of Asian in controls:\n')
        stat_file.write(str(num_Asian_nonmci))
        stat_file.write('\n')
        stat_file.write('percentage of Asian in controls:\n')
        stat_file.write(str(perc_Asian_nonmci))
        stat_file.write('\n')

        num_Black_nonmci = nonmci_metadata_cohort[nonmci_metadata_cohort['canonical_race'] == 'Black'].shape[0]
        perc_Black_nonmci =  num_Black_nonmci/nonmci_metadata_cohort.shape[0]   
        stat_file.write('number of Black in controls:\n')
        stat_file.write(str(num_Black_nonmci))
        stat_file.write('\n')
        stat_file.write('percentage of Black in controls:\n')
        stat_file.write(str(perc_Black_nonmci))
        stat_file.write('\n')

        num_Unknown_nonmci = nonmci_metadata_cohort[nonmci_metadata_cohort['canonical_race'] == 'Unknown'].shape[0]
        perc_Unknown_nonmci =  num_Unknown_nonmci/nonmci_metadata_cohort.shape[0]   
        stat_file.write('number of Unknown in controls:\n')
        stat_file.write(str(num_Unknown_nonmci))
        stat_file.write('\n')
        stat_file.write('percentage of Unknown in controls:\n')
        stat_file.write(str(perc_Unknown_nonmci))
        stat_file.write('\n')

        num_Pacific_Islander_nonmci = nonmci_metadata_cohort[nonmci_metadata_cohort['canonical_race'] == 'Pacific Islander'].shape[0]
        perc_Pacific_Islander_nonmci =  num_Pacific_Islander_nonmci/nonmci_metadata_cohort.shape[0]   
        stat_file.write('number of Pacific Islander in controls:\n')
        stat_file.write(str(num_Pacific_Islander_nonmci))
        stat_file.write('\n')
        stat_file.write('percentage of Pacific Islander in controls:\n')
        stat_file.write(str(perc_Pacific_Islander_nonmci))
        stat_file.write('\n')

        num_Native_American_nonmci = nonmci_metadata_cohort[nonmci_metadata_cohort['canonical_race'] == 'Native American'].shape[0]
        perc_Native_American_nonmci =  num_Native_American_nonmci/nonmci_metadata_cohort.shape[0]   
        stat_file.write('number of Native American in controls:\n')
        stat_file.write(str(num_Native_American_nonmci))
        stat_file.write('\n')
        stat_file.write('percentage of Native American in controls:\n')
        stat_file.write(str(perc_Native_American_nonmci))
        stat_file.write('\n')

        # pdb.set_trace()
        # Stats on important features
        feature_ranking = pd.read_csv(feature_ranking_path, names=['Feature', 'Score']).sort_values(by='Score', ascending=False)
        feature_ranking['Feature'] = feature_ranking['Feature'].str.strip()
        feature_ranking = feature_ranking[~feature_ranking['Feature'].isin(['age from bdate to 2022','race','sex'])]

        for i in range(30):
            important_feature = feature_ranking['Feature'].iloc[i]
            num_mcipatients = sum(data_all_pos[important_feature] > 0)
            perc_mcipatients = num_mcipatients/(data_all_pos.shape[0])
            
            num_nonmcipatients = sum(data_all_neg[important_feature] > 0)
            perc_nonmcipatients = num_nonmcipatients/(data_all_neg.shape[0])

            stat_file.write('Feature number '+str(i)+':\n')
            stat_file.write(important_feature)
            stat_file.write('\n')
            stat_file.write('Number of patients in cases:\n')
            stat_file.write(str(num_mcipatients))
            stat_file.write('\n')
            stat_file.write('Percentage of patients in cases:\n')
            stat_file.write(str(perc_mcipatients))
            stat_file.write('\n')
            stat_file.write('Number of patients in controls:\n')
            stat_file.write(str(num_nonmcipatients))
            stat_file.write('\n')
            stat_file.write('Percentage of patients in controls:\n')
            stat_file.write(str(perc_nonmcipatients))
            stat_file.write('\n')        
        pdb.set_trace()
        print('Test')

