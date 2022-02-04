import pdb
import pandas as pd
import itertools
import math
import collections
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

def blind_data(line_med_splited
							, line_diag_splited
							, line_proc_splited
							, current_patient_demog
							, prediction_window_size):
	# pdb.set_trace()
	line_meds_blinded = []
	line_diags_blinded = []
	line_procs_blinded = []
	first_record_date = min(datetime.strptime(line_med_splited[0][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d'), datetime.strptime(line_proc_splited[0][0][:10], '%Y-%m-%d'))	
	last_record_date = min(datetime.strptime(line_med_splited[-1][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d'), datetime.strptime(line_proc_splited[-1][0][:10], '%Y-%m-%d'))
	if current_patient_demog['MCI_label'].values[0] == 1:
		idx_date = datetime.strptime(current_patient_demog['diag_date'].values[0][:10],'%Y-%m-%d')
	else:
		idx_date = last_record_date + relativedelta(months= -prediction_window_size)
		
	for i in range(len(line_med_splited)):
		current_date = datetime.strptime(line_med_splited[i][0][:10], '%Y-%m-%d') 
		first_to_idx_date =  (idx_date.year - first_record_date.year) * 12 + idx_date.month - first_record_date.month 
		if first_to_idx_date >= prediction_window_size:
			line_meds_blinded.append(line_med_splited[i])

	for i in range(len(line_diag_splited)):
		current_date = datetime.strptime(line_diag_splited[i][0][:10], '%Y-%m-%d') 
		first_to_idx_date =  (idx_date.year - first_record_date.year) * 12 + idx_date.month - first_record_date.month 
		if first_to_idx_date >= prediction_window_size:
			line_diags_blinded.append(line_diag_splited[i])

	for i in range(len(line_proc_splited)):
		current_date = datetime.strptime(line_proc_splited[i][0][:10], '%Y-%m-%d') 
		first_to_idx_date =  (idx_date.year - first_record_date.year) * 12 + idx_date.month - first_record_date.month 
		if first_to_idx_date >= prediction_window_size:
			line_procs_blinded.append(line_proc_splited[i])

	return line_meds_blinded, line_diags_blinded, line_procs_blinded	

def check_data_availibility(line_med_splited
							, line_diag_splited
							, line_proc_splited
							, current_patient_demog
							, prediction_window_size):
	# Check data availibility
	# pdb.set_trace()
	elig_flag = True
	first_record_date = min(datetime.strptime(line_med_splited[0][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d'), datetime.strptime(line_proc_splited[0][0][:10], '%Y-%m-%d'))
	last_record_date = min(datetime.strptime(line_med_splited[-1][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d'), datetime.strptime(line_proc_splited[-1][0][:10], '%Y-%m-%d'))
	if current_patient_demog['MCI_label'].values[0] == 1:		
		diag_date = datetime.strptime(current_patient_demog['diag_date'].values[0][:10],'%Y-%m-%d')
		first_to_idx_date =  (diag_date.year - first_record_date.year) * 12 + diag_date.month - first_record_date.month 
		idx_date_to_last = (last_record_date.year - diag_date.year ) * 12 + last_record_date.month - diag_date.month 
	else:
		diag_date = 'NA'
		idx_date = last_record_date + relativedelta(months= -prediction_window_size)
		first_to_idx_date =  (idx_date.year - first_record_date.year) * 12 + idx_date.month - first_record_date.month
		idx_date_to_last = (last_record_date.year - idx_date.year ) * 12 + last_record_date.month - idx_date.month 
	if (first_to_idx_date >= (2*prediction_window_size)) and (idx_date_to_last >= prediction_window_size):
		elig_flag = True 
	else:
		elif_flag = False	
	# pdb.set_trace()
	return elig_flag



def create_stationary_icd10(line_diag_splited, frequent_icd10s_dict):
	# pdb.set_trace()
	epsil = 2.220446049250313e-16
	round_num_digit = 5
	for i in range(len(line_diag_splited)):
		# if line_diag_splited[i][0]
		for j in range(1, len(line_diag_splited[i])): 
			if (line_diag_splited[i][j][:5] == 'ICD10') and (line_diag_splited[i][j][6:] in frequent_icd10s_dict):           
				frequent_icd10s_dict[ line_diag_splited[i][j][6:]] += 1
			else:
				continue  
				print('test') 
	# pdb.set_trace()
	frequent_icd10s_dict_sorted = dict(collections.OrderedDict(sorted(frequent_icd10s_dict.items())))    
	num_records = len(line_diag_splited)
	if (num_records > 1):
		frequent_icd10s_dict_sorted = {k: np.round(v / (math.log(num_records, 2)+ epsil), round_num_digit) for k, v in frequent_icd10s_dict_sorted.items()}
	else:
		frequent_icd10s_dict_sorted = {k: np.round(v , round_num_digit) for k, v in frequent_icd10s_dict_sorted.items()}        
	# pdb.set_trace()
	return frequent_icd10s_dict_sorted

def create_stationary_icd9(line_diag_splited, frequent_icd9s_dict):
	# pdb.set_trace()
	epsil = 2.220446049250313e-16
	round_num_digit = 5
	for i in range(len(line_diag_splited)):
		# if line_diag_splited[i][0]
		for j in range(1, len(line_diag_splited[i])): 
			if (line_diag_splited[i][j][:4] == 'ICD9') and (line_diag_splited[i][j][5:] in frequent_icd9s_dict):           
				frequent_icd9s_dict[ line_diag_splited[i][j][5:]] += 1
			else:
				continue  
				print('test') 
	# pdb.set_trace()
	frequent_icd9s_dict_sorted = dict(collections.OrderedDict(sorted(frequent_icd9s_dict.items())))    
	num_records = len(line_diag_splited)
	if (num_records > 1):
		frequent_icd9s_dict_sorted = {k: np.round(v / (math.log(num_records, 2)+ epsil), round_num_digit) for k, v in frequent_icd9s_dict_sorted.items()}
	else:
		frequent_icd9s_dict_sorted = {k: np.round(v , round_num_digit) for k, v in frequent_icd9s_dict_sorted.items()}        
	# pdb.set_trace()
	return frequent_icd9s_dict_sorted

def create_stationary_med(line_med_splited, frequent_meds_dict):
	# pdb.set_trace()
	epsil = 2.220446049250313e-16
	round_num_digit = 5
	for i in range(len(line_med_splited)):
		# if line_med_splited[i][0]
		for j in range(1, len(line_med_splited[i])): 
			if (line_med_splited[i][j] in frequent_meds_dict):           
				frequent_meds_dict[ line_med_splited[i][j]] += 1
			else:
				continue  
				print('test') 
	# pdb.set_trace()
	frequent_meds_dict_sorted = dict(collections.OrderedDict(sorted(frequent_meds_dict.items())))    
	num_records = len(line_med_splited)
	if (num_records > 1):
		frequent_meds_dict_sorted = {k: np.round(v / (math.log(num_records, 2)+ epsil), round_num_digit) for k, v in frequent_meds_dict_sorted.items()}
	else:
		frequent_meds_dict_sorted = {k: np.round(v , round_num_digit) for k, v in frequent_meds_dict_sorted.items()}        
	# pdb.set_trace()
	return frequent_meds_dict_sorted

def create_stationary_proc(line_proc_splited, frequent_procs_dict):
	# pdb.set_trace()
	epsil = 2.220446049250313e-16
	round_num_digit = 5
	for i in range(len(line_proc_splited)):
		# if line_proc_splited[i][0]
		for j in range(1, len(line_proc_splited[i])): 
			if (line_proc_splited[i][j] in frequent_procs_dict):           
				frequent_procs_dict[ line_proc_splited[i][j]] += 1
			else:
				continue  
				print('test') 
	# pdb.set_trace()
	frequent_procs_dict_sorted = dict(collections.OrderedDict(sorted(frequent_procs_dict.items())))    
	num_records = len(line_proc_splited)
	if (num_records > 1):
		frequent_procs_dict_sorted = {k: np.round(v / (math.log(num_records, 2)+ epsil), round_num_digit) for k, v in frequent_procs_dict_sorted.items()}
	else:
		frequent_procs_dict_sorted = {k: np.round(v , round_num_digit) for k, v in frequent_procs_dict_sorted.items()}        
	# pdb.set_trace()
	return frequent_procs_dict_sorted

def create_stationary(diagnosis_file_path
					, medication_file_path
					, procedure_file_path
					, icd10_frequencies_mci_path
					, icd9_frequencies_mci_path
					, medication_id_frequencies_mci_path
					, procedure_id_frequencies_mci_path
					, demographic_file_path
					, top_n		
					, prediction_window_size
					, cohort			
					):
	# pdb.set_trace()
	demographic_data = pd.read_csv(demographic_file_path)
	demographic_data['sex'] = demographic_data['sex'].map({'Male': 1, 'Female':2, 'Unknown': 0})
	demographic_data['canonical_race'] = demographic_data['canonical_race'].map({'Native American': 1, 'Black':2, 'White': 3, 'Pacific Islander':4, 'Asian':5, 'Unknown':6, 'Other':7})

	
	# Rread frequent ICD codes
	frequent_icd10s = pd.read_csv(icd10_frequencies_mci_path)
	frequent_icd10s.columns=frequent_icd10s.columns.str.strip()
	frequent_icd10s = frequent_icd10s.sort_values('num patient', ascending=False)
	frequent_icd10s_top_n = frequent_icd10s['Code'].values[:top_n].tolist()

	frequent_icd9s = pd.read_csv(icd9_frequencies_mci_path)
	frequent_icd9s.columns=frequent_icd9s.columns.str.strip()
	frequent_icd9s = frequent_icd9s.sort_values('num patient', ascending=False)
	frequent_icd9s_top_n = frequent_icd9s['Code'].values[:top_n].tolist()

	frequent_meds = pd.read_csv(medication_id_frequencies_mci_path)
	frequent_meds.columns=frequent_meds.columns.str.strip()
	frequent_meds = frequent_meds.sort_values('num patient', ascending=False)
	frequent_meds_top_n = frequent_meds['Code'].values[:top_n].tolist()

	frequent_procs = pd.read_csv(procedure_id_frequencies_mci_path )
	frequent_procs.columns = frequent_procs.columns.str.strip()
	frequent_procs = frequent_procs.sort_values('num patient', ascending=False)	
	frequent_procs_top_n = frequent_procs['Code'].values[:top_n].tolist()


	frequent_icd10s_dict = {}
	for i in range(len(frequent_icd10s_top_n)):
		frequent_icd10s_dict[frequent_icd10s_top_n[i]] = 0 

	frequent_icd9s_dict = {}
	for i in range(len(frequent_icd9s_top_n)):
		frequent_icd9s_dict[frequent_icd9s_top_n[i]] = 0 

	frequent_meds_dict = {}
	for i in range(len(frequent_meds_top_n)):
		frequent_meds_dict[frequent_meds_top_n[i]] = 0 

	frequent_procs_dict = {}
	for i in range(len(frequent_procs_top_n)):
		frequent_procs_dict[frequent_procs_top_n[i]] = 0 


	# pdb.set_trace()
	with open(diagnosis_file_path) as diag_file, open(medication_file_path) as med_file, open(procedure_file_path) as proc_file, open('stationary_data/stationary_dataset_'+cohort+'.csv', 'w') as stationary_file, open('intermediate_files/elig_patients_'+cohort+'.csv', 'w') as elig_patients_file, open('intermediate_files/unelig_patients_'+cohort+'.csv', 'w') as unelig_patients_file:
		header_diag = next(diag_file)
		header_med = next(med_file)
		header_proc = next(proc_file)

		elig_patients_file.write('eligible_patients_ids\n')
		unelig_patients_file.write('uneligible_patients_ids\n')

		stationary_file.write('Patient_ID, sex, race, age from bdate to 2022, '+ (','.join([*frequent_icd10s_dict.keys()])) )
		stationary_file.write(',')

		stationary_file.write(','.join([*frequent_icd9s_dict.keys()]))
		stationary_file.write(',')       

		stationary_file.write(','.join(str(x) for x in [*frequent_meds_dict.keys()]))
		stationary_file.write(',')  

		stationary_file.write(','.join(str(x) for x in [*frequent_procs_dict.keys()]))
		stationary_file.write(',') 
		stationary_file.write('Label')    
		stationary_file.write('\n')       
		line_counter = 0
		for line in diag_file:
			line_counter+=1
			line_diag = line.replace('\n','').split(',')
			line_diag_splited = [list(y) for x, y in itertools.groupby(line_diag[1:], lambda z: z == 'EOV') if not x]    
	
			line_med = med_file.readline().replace('\n','').split(',')
			line_med_splited = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]    

			line_proc = proc_file.readline().replace('\n','').split(',')
			line_proc_splited = [list(y) for x, y in itertools.groupby(line_proc[1:], lambda z: z == 'EOV') if not x]    
			
			if not(line_diag[0] == line_med[0] == line_proc[0]):
				pdb.set_trace()
				break
				print('Patients dont match')
			# line_demog = demogs_filename.readline().rstrip('\n')
			# line_demog = line_demog.split(',')  
			# pdb.set_trace()
			# Note, index time for MCI cohort is diagnosis date and for non-MCI cohort is last record date - T months
			# MCI positive patients should have diagnosis date-2T months (from data first record date + 2T < diagnosis date) of data.
			# MCI negative patients should have at least 3T months of data
			# Blind the data for both MCI (diagnosis date - T months) and fot non-MCI patients (from last record data - 2T to last record date - T)
			current_patient_demog = demographic_data[demographic_data['anon_id']==line_med[0]]
			
			elig_flag = check_data_availibility(line_med_splited
													, line_diag_splited
													, line_proc_splited
													, current_patient_demog
													, prediction_window_size)

			if elig_flag == True:
				elig_patients_file.write(line_med[0])
				elig_patients_file.write('\n')
			else:	
				unelig_patients_file.write(line_med[0])
				unelig_patients_file.write('\n')
				continue

			line_med_blinded, line_diag_blinded, line_proc_blinded= blind_data(line_med_splited
									, line_diag_splited
									, line_proc_splited
									, current_patient_demog
									, prediction_window_size)


			frequent_icd10s_dict = dict.fromkeys(frequent_icd10s_dict, 0)    
			frequent_icd9s_dict = dict.fromkeys(frequent_icd9s_dict, 0)    
			frequent_meds_dict = dict.fromkeys(frequent_meds_dict, 0)    
			frequent_procs_dict = dict.fromkeys(frequent_procs_dict, 0)    
			
			stationary_icd10_vect = create_stationary_icd10(line_diag_blinded, frequent_icd10s_dict)		
			stationary_icd9_vect = create_stationary_icd9(line_diag_blinded, frequent_icd9s_dict)		
			stationary_med_vect = create_stationary_med(line_med_blinded, frequent_meds_dict)					
			stationary_proc_vect = create_stationary_proc(line_proc_blinded, frequent_procs_dict)		
			
			print(line_counter)
			# pdb.set_trace()
			stationary_file.write((line_med[0]))
			stationary_file.write(',')

			stationary_file.write(str(current_patient_demog['sex'].values[0]))
			stationary_file.write(',')
			stationary_file.write(str(current_patient_demog['canonical_race'].values[0]))
			stationary_file.write(',')
			stationary_file.write(str(2022-int(current_patient_demog['bdate'].values[0].split('-')[0])))
			stationary_file.write(',')


			# pdb.set_trace()
			stationary_file.write(','.join(map(repr, list(stationary_icd10_vect.values()))))
			stationary_file.write(',')
			stationary_file.write(','.join(map(repr, list(stationary_icd9_vect.values()))))
			stationary_file.write(',')
			stationary_file.write(','.join(map(repr, list(stationary_med_vect.values()))))
			stationary_file.write(',')
			stationary_file.write(','.join(map(repr, list(stationary_proc_vect.values()))))
			stationary_file.write(',')
			stationary_file.write(str(current_patient_demog['MCI_label'].values[0]))
			stationary_file.write('\n')            

			print('End')
