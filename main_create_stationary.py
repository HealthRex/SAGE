'''
1. num_features: Read frequency files for medications, diagnosis and procedure select top-n features. 
2. For each patient in medication file:
	a. Read their medication
	b. Iterate through their timestamps.
	c. Form a m by n matrix where m is the number of timestamps and n is the top-n features
3 Repeat step 2 for diagnosis and procedure

'''
import utils.create_stationary as sta
import argparse
import os
import pdb
import sys

# pdb.set_trace()
sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--cohort", type=str, default='mci', choices = ['mci', 'non_mci'])    
parser.add_argument("--top_n", type=int, default=10)    
parser.add_argument("--prediction_window_size", type=int, default=6)    


parser.add_argument("--mci_diagnosis_file_path", type=str, default='intermediate_files/diagnosis_codes_mci.csv')    
parser.add_argument("--mci_medication_file_path", type=str, default='intermediate_files/medication_codes_mci.csv')    
parser.add_argument("--mci_procedure_file_path", type=str, default='intermediate_files/procedure_codes_mci.csv')    
parser.add_argument("--mci_demographic_file_path", type=str, default='intermediate_files/mci_all_metadata.csv')    

parser.add_argument("--non_mci_diagnosis_file_path", type=str, default='intermediate_files/diagnosis_codes_non_mci.csv')    
parser.add_argument("--non_mci_medication_file_path", type=str, default='intermediate_files/medication_codes_non_mci.csv')    
parser.add_argument("--non_mci_procedure_file_path", type=str, default='intermediate_files/procedure_codes_non_mci.csv')    
parser.add_argument("--non_mci_demographic_file_path", type=str, default='intermediate_files/non_mci_all_visited_neurology_metadata.csv')    


parser.add_argument("--icd10_frequencies_mci_path", type=str, default='intermediate_files/icd10_frequencies_mci.csv')    
parser.add_argument("--icd9_frequencies_mci_path", type=str, default='intermediate_files/icd9_frequencies_mci.csv')    
parser.add_argument("--medication_id_frequencies_mci_path", type=str, default='intermediate_files/medication_id_frequencies_mci.csv')    
parser.add_argument("--procedure_id_frequencies_mci_path", type=str, default='intermediate_files/procedure_id_frequencies_mci.csv')    


if parser.parse_args().cohort == 'mci':
	sta.create_stationary(parser.parse_args().mci_diagnosis_file_path
							,parser.parse_args().mci_medication_file_path
							,parser.parse_args().mci_procedure_file_path
							,parser.parse_args().icd10_frequencies_mci_path
							,parser.parse_args().icd9_frequencies_mci_path
							,parser.parse_args().medication_id_frequencies_mci_path
							,parser.parse_args().procedure_id_frequencies_mci_path
							,parser.parse_args().mci_demographic_file_path
							,parser.parse_args().top_n							 
							,parser.parse_args().prediction_window_size							 							
							,parser.parse_args().cohort
							)
							 
if parser.parse_args().cohort == 'non_mci':
	sta.create_stationary(parser.parse_args().non_mci_diagnosis_file_path
							,parser.parse_args().non_mci_medication_file_path
							,parser.parse_args().non_mci_procedure_file_path
							,parser.parse_args().icd10_frequencies_mci_path
							,parser.parse_args().icd9_frequencies_mci_path
							,parser.parse_args().medication_id_frequencies_mci_path
							,parser.parse_args().procedure_id_frequencies_mci_path
							,parser.parse_args().non_mci_demographic_file_path
							,parser.parse_args().top_n	
							,parser.parse_args().prediction_window_size							 															
							,parser.parse_args().cohort												 
							)
