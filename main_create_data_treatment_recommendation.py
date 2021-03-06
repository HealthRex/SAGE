import pdb
import argparse
import sys
import os
import utils.create_stationary_treatment_recom as trt_recom
import logging


sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()

parser.add_argument("--treatment_window_days", type=int, default=30)    
parser.add_argument("--mci_metadata_path", type=str, default='intermediate_files/mci_metadata.csv')    
parser.add_argument("--data_path", type=str, default='stationary_data/for_recommender_stationary_dataset_mci.csv')    
parser.add_argument("--mci_procedure_codes_path", type=str, default='intermediate_files/procedure_codes_mci.csv')    
parser.add_argument("--procedure_id_frequencies_mci_path", type=str, default='dict_files/frequent_procs_desc.csv')    
parser.add_argument("--top_n_proc", type=int, default=100)    
parser.add_argument("--test_ratio", type=int, default=0.3)    



args = parser.parse_args()


trt_recom.create_stationary_for_treatment_recom(args.mci_metadata_path
											, args.data_path
											# , args.test_data_path
											, args.mci_procedure_codes_path
											, args.procedure_id_frequencies_mci_path
											, args.treatment_window_days
											, args.top_n_proc
											, args.test_ratio)


