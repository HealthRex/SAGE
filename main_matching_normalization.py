'''

'''
import pdb
import argparse
import os
import sys
import utils.matching_normalization as mt_norm

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--mci_stationary_data_path", type=str, default='stationary_data/stationary_dataset_mci.csv')    
parser.add_argument("--non_mci_stationary_data_path", type=str, default='stationary_data/stationary_dataset_non_mci.csv')    

parser.add_argument("--mci_metadata_path", type=str, default='intermediate_files/mci_all_metadata.csv')    
parser.add_argument("--non_mci_metadata_path", type=str, default='intermediate_files/non_mci_all_visited_neurology_metadata.csv')    

parser.add_argument("--matching", type=int, default=1, choices=[0,1])    
parser.add_argument("--case_control_ratio", type=int, default=1)    

parser.add_argument("--test_ratio", type=int, default=0.3)    

output_path = mt_norm.matching(parser.parse_args().mci_stationary_data_path
				,parser.parse_args().non_mci_stationary_data_path
				,parser.parse_args().mci_metadata_path
				,parser.parse_args().non_mci_metadata_path								
				,parser.parse_args().case_control_ratio												
				,parser.parse_args().matching												
				)

mt_norm.normalization(output_path
					,parser.parse_args().test_ratio
					)