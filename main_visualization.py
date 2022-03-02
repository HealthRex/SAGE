import pdb
import argparse
import sys
import os
import viz.viz_functions as vis_tools


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()

parser.add_argument("--viz_method", type=str, default="none", choices = ["pca", "tsne", "none"])    
parser.add_argument("--features_to_show", type=str, default="meds_diags_procs", choices = ["meds_diags_procs","all"])    
parser.add_argument("--sampled", type=int, default=1, choices = [0,1])    
parser.add_argument("--sample_size", type=int, default=2000)    
parser.add_argument("--perplex", type=int, default=15)    
parser.add_argument("--num_it", type=int, default=2000)    
parser.add_argument("--lr_rate", type=int, default=200)   

parser.add_argument("--sample_size_for_shap", type=float, default=0.05)  
parser.add_argument("--trained_model_path", type=str, default="saved_classical_ml_models/rf_model.pkl")    


parser.add_argument("--compute_table_1", type=int, default=0, choices = [0, 1])    


parser.add_argument("--train_stationary_filename", type=str, default="stationary_data/stationary_data_imbratio1_normalized_train.csv")    
parser.add_argument("--test_stationary_filename", type=str, default="stationary_data/stationary_data_imbratio1_normalized_test.csv")    
parser.add_argument("--feature_ranking_path", type=str, default="saved_classical_ml_models/feature_impoerance_rf.csv")    

parser.add_argument("--mci_metadata", type=str, default="intermediate_files/mci_metadata.csv")  
parser.add_argument("--nonmci_metadata", type=str, default="intermediate_files/nonmci_metadata.csv")    


if parser.parse_args().viz_method == "tsne":
    args = parser.parse_args()
    vis_tools.tSNE_visualization(args.train_stationary_filename 
                                , args.test_stationary_filename     
                                , args.sampled     
                                , args.sample_size
                                , args.features_to_show
                                , args.perplex
                                , args.num_it
                                , args.lr_rate)
  
elif parser.parse_args().viz_method == "pca":
    args = parser.parse_args()
    vis_tools.pca_visualization(args.train_stationary_filename 
                                , args.test_stationary_filename     
                                , args.sampled     
                                , args.sample_size
                                , args.features_to_show
                                )   
elif parser.parse_args().viz_method == "none":
    print("Warning: no visualization method has been selected.")



if parser.parse_args().compute_table_1 == 1:
    args = parser.parse_args()
    vis_tools.compute_table_stats(args.train_stationary_filename 
                                , args.test_stationary_filename     
                                , args.features_to_show
                                , args.mci_metadata     
                                , args.nonmci_metadata                                
                                , args.feature_ranking_path                                
                                )
  
elif parser.parse_args().compute_table_1 == 0:
    print("Warning: no compute_table_1 method has been selected.")







