'''
1. num_features: Read frequency files for medications, diagnosis and procedure select top-n features. 
2. For each patient in medication file:
	a. Read their medication
	b. Iterate through their timestamps.
	c. Form a m by n matrix where m is the number of timestamps and n is the top-n features
3 Repeat step 2 for diagnosis and procedure

'''
import utils.create_longitudinal as longit
import argparse
import os
import pdb
import sys
import logging


# pdb.set_trace()
sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()

parser.add_argument("--logging_milestone", type=int, default=1000)    
# logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.INFO, filename = 'log/logfile_create_stationary.log', filemode = 'a')


parser.add_argument("--cohort", type=str, default='mci', choices = ['mci', 'nonmci'])    
parser.add_argument("--top_n_med", type=int, default=100)    
parser.add_argument("--top_n_proc", type=int, default=100)    
parser.add_argument("--top_n_ccs", type=int, default=100)    

parser.add_argument("--prediction_window_size", type=int, default=2)    
parser.add_argument("--treatment_window_days", type=int, default=30)    

parser.add_argument("--recommender", type=str, default='for_recommender_', choices=['', 'for_recommender_'])    



parser.add_argument("--mci_diagnosis_file_path", type=str, default='intermediate_files/diagnosis_codes_mci.csv')    
parser.add_argument("--mci_medication_file_path", type=str, default='intermediate_files/medication_codes_mci.csv')    
parser.add_argument("--mci_procedure_file_path", type=str, default='intermediate_files/procedure_codes_mci.csv')    
parser.add_argument("--mci_demographic_file_path", type=str, default='intermediate_files/mci_metadata.csv')    

parser.add_argument("--non_mci_diagnosis_file_path", type=str, default='intermediate_files/diagnosis_codes_nonmci.csv')    
parser.add_argument("--non_mci_medication_file_path", type=str, default='intermediate_files/medication_codes_nonmci.csv')    
parser.add_argument("--non_mci_procedure_file_path", type=str, default='intermediate_files/procedure_codes_nonmci.csv')    
parser.add_argument("--non_mci_demographic_file_path", type=str, default='intermediate_files/nonmci_metadata.csv')    


parser.add_argument("--ccs_frequencies_mci_path", type=str, default='intermediate_files/ccs_frequencies_mci.csv')    
parser.add_argument("--pharm_class_frequencies_mci_path", type=str, default='intermediate_files/pharm_class_frequencies_mci.csv')    
parser.add_argument("--procedure_id_frequencies_mci_path", type=str, default='intermediate_files/procedure_id_frequencies_mci.csv')    


parser.add_argument("--stationary_train_data_path", type=str, default='recommender_data/recomender_data_train.csv')    
parser.add_argument("--stationary_test_data_path", type=str, default='recommender_data/recomender_data_test.csv')    

if parser.parse_args().cohort == 'mci':
	longit.create_longitudinal(parser.parse_args().mci_diagnosis_file_path
							,parser.parse_args().mci_medication_file_path
							,parser.parse_args().mci_procedure_file_path
							,parser.parse_args().ccs_frequencies_mci_path
							,parser.parse_args().pharm_class_frequencies_mci_path
							,parser.parse_args().procedure_id_frequencies_mci_path
							,parser.parse_args().mci_demographic_file_path							 
							,parser.parse_args().top_n_ccs							 
							,parser.parse_args().top_n_med
							,parser.parse_args().top_n_proc							 
							,parser.parse_args().prediction_window_size							 							
							,parser.parse_args().cohort
							,parser.parse_args().logging_milestone
							,parser.parse_args().recommender
							,parser.parse_args().treatment_window_days
							,parser.parse_args().stationary_train_data_path
							,parser.parse_args().stationary_test_data_path
							)
							 

