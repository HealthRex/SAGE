import pdb
import pandas as pd
import itertools
import math
import collections
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import csv 


def create_stationary_for_treatment_recom(mci_metadata_path
											, train_data_path
											, test_data_path
											, mci_procedure_codes_path
											, procedure_id_frequencies_mci_path
											, treatment_window
											, top_n_proc):
	'''
	Read mci metadata. 
	For each mci patient:
		extract their stationary vector that you have already created earlier for the prediction task.
		Extract their procedure codes from intermediate_files/procedure_codes_mci.csv.
		Select the procedure codes within MCI diagnosis date +/- 1 month and create a multi-hot encoding vector.
		To create the multihot-encoding vector
	'''	
	# pdb.set_trace()
	mci_metadata = pd.read_csv(mci_metadata_path)
	train_data = pd.read_csv(train_data_path)
	test_data = pd.read_csv(test_data_path)

	train_pos = train_data[train_data['Label']==1]
	test_pos = test_data[test_data['Label']==1]

	mci_data = pd.concat([train_pos, test_pos])

	frequent_procs = pd.read_csv(procedure_id_frequencies_mci_path, dtype=str)
	frequent_procs.columns = frequent_procs.columns.str.strip()
	frequent_procs = frequent_procs.sort_values('num patient', ascending=False)	
	frequent_procs_top_n = frequent_procs['Code'].values[:top_n_proc].tolist()
	frequent_procs_dict = {}
	for i in range(len(frequent_procs_top_n)):
		frequent_procs_dict[frequent_procs_top_n[i]] = 0 
	frequent_procs_dict = dict(collections.OrderedDict(sorted(frequent_procs_dict.items())))   

	with open(mci_procedure_codes_path) as proc_file, open('recommender_data/recomender_data_train.csv','w') as recom_train_file, open('recommender_data/recomender_data_test.csv','w') as recom_test_file:
		proc_file_header = next(proc_file)
		recom_train_file.write('Patient_ID')
		recom_train_file.write(',')
		recom_train_file.write(','.join(mci_data.columns[:-2].tolist()))
		recom_train_file.write(',')
		recom_train_file.write(','.join('target_proc_'+str(x) for x in [*frequent_procs_dict.keys()]))
		recom_train_file.write('\n')

		recom_test_file.write('Patient_ID')
		recom_test_file.write(',')
		recom_test_file.write(','.join(mci_data.columns[:-2].tolist()))
		recom_test_file.write(',')
		recom_test_file.write(','.join('target_proc_'+str(x) for x in [*frequent_procs_dict.keys()]))
		recom_test_file.write('\n')		

		for line in proc_file:			
			line=line.split(',')
			line_proc_splited = [list(y) for x, y in itertools.groupby(line[1:], lambda z: z == 'EOV') if not x]    

			current_patient_id = line[0]

			current_patient_metadata = mci_metadata[mci_metadata['anon_id']==current_patient_id]
			if current_patient_metadata.shape[0] <1:
				pdb.set_trace()
				print('Warning: metadata was not found for this patient')

			current_patient_stat_data = mci_data[mci_data['Patient_ID']==current_patient_id]	
			if current_patient_stat_data.shape[0] < 1:
				continue
				print('Stationary vector was not found for the current patient')

			current_diag_datetime = current_patient_metadata['index_date_OR_diag_date'].values[0]
			current_diag_date = datetime.strptime(current_diag_datetime[:10], '%Y-%m-%d')
			
			current_patient_procs = []
			frequent_procs_dict = dict.fromkeys(frequent_procs_dict, 0) 
			for j in range(len(line_proc_splited)):
				current_vdate = datetime.strptime(line_proc_splited[j][0], '%Y-%m-%d')
				if (current_vdate < (current_diag_date + relativedelta(days= treatment_window))) and (current_vdate > (current_diag_date + relativedelta(days= -treatment_window))):
					current_patient_procs.append(line_proc_splited[j])
					# print('Test')
			# pdb.set_trace()	
			current_patient_procs_flatten = [item for sublist in current_patient_procs for item in sublist[1:]]

			for j in range(len(current_patient_procs_flatten)): 
				if (current_patient_procs_flatten[j] in frequent_procs_dict):           
					frequent_procs_dict[current_patient_procs_flatten[j]] = 1
			
			frequent_procs_dict_sorted = dict(collections.OrderedDict(sorted(frequent_procs_dict.items())))    
			
			# pdb.set_trace()		
			if current_patient_stat_data['Patient_ID'].values[0] in train_pos['Patient_ID'].values.tolist():
				# pdb.set_trace()
				recom_train_file.write(str(current_patient_stat_data['Patient_ID'].values[0]))
				recom_train_file.write(',')
				recom_train_file.write(','.join(map(repr, current_patient_stat_data.values.tolist()[0][:-2])))
				recom_train_file.write(',')
				recom_train_file.write(','.join(map(repr, list(frequent_procs_dict_sorted.values()))))
				recom_train_file.write('\n')
			elif current_patient_stat_data['Patient_ID'].values[0] in test_pos['Patient_ID'].values.tolist():
				recom_test_file.write(str(current_patient_stat_data['Patient_ID'].values[0]))
				recom_test_file.write(',')				
				recom_test_file.write(','.join(map(repr, current_patient_stat_data.values.tolist()[0][:-2])))
				recom_test_file.write(',')
				recom_test_file.write(','.join(map(repr, list(frequent_procs_dict_sorted.values()))))
				recom_test_file.write('\n')
			else:
				pdb.set_trace()
				print('Patient not found')	


