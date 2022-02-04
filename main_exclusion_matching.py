'''

'''
import pdb
import argparse
import os
import sys
import utils.exclusion_matching as ex_mt

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--cohort", type=str, default='mci', choices = ['mci', 'non_mci'])    
parser.add_argument("--mci_stationary_data_path", type=str, default='stationary_data/stationary_dataset_mci.csv')    
parser.add_argument("--prediction_window_size", type=int, default=6)    
parser.add_argument("--prior_clean_window_size", type=int, default=6)    


if parser.parse_args().cohort == 'mci':
	ex_mt.apply_exclusion_criteria(parser.parse_args().cohort
								,parser.parse_args().mci_stationary_data_path
								,parser.parse_args().prediction_window_size
								,parser.parse_args().prior_clean_window_size								
		)
