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

logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.INFO, filename = 'log/logfile_create_longitudinal.log', filemode = 'a')


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def vdates_syncher(current_patient_vdates):
    # pdb.set_trace()
    current_patient_vdates_synched = []
    current_patient_vdates_synched.append(int(current_patient_vdates[0]))
    for i in range(1, len(current_patient_vdates)):
        current_vdate= int(current_patient_vdates[i])
        prev_vdate= int(current_patient_vdates[i-1])
        diff_times = diff_month(datetime(current_vdate//100,current_vdate%100, 1 ), datetime(prev_vdate//100,prev_vdate%100, 1))
        if diff_times <2:
            current_patient_vdates_synched.append(current_vdate)
        else:
            for j in range(diff_times-1):   
                # pdb.set_trace()
                current_patient_vdates_synched.append(int((datetime(prev_vdate//100, prev_vdate%100,1) + relativedelta(months=+1)).strftime("%Y%m")))
                prev_vdate=current_patient_vdates_synched[-1]
            current_patient_vdates_synched.append(current_vdate)
    return  current_patient_vdates_synched 

def blind_data(patient_id
				,line_med_splited
				, line_diag_splited
				, line_proc_splited
				, current_patient_demog
				, prediction_window_size):
	line_meds_blinded = []
	line_diags_blinded = []
	line_procs_blinded = []
	if (line_med_splited != [] and line_diag_splited != [] and line_proc_splited != []):
		first_record_date = min(datetime.strptime(line_med_splited[0][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d'), datetime.strptime(line_proc_splited[0][0][:10], '%Y-%m-%d'))
		last_record_date = max(datetime.strptime(line_med_splited[-1][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d'), datetime.strptime(line_proc_splited[-1][0][:10], '%Y-%m-%d'))
	elif (line_med_splited != [] and line_diag_splited != [] and line_proc_splited == []):
		first_record_date = min(datetime.strptime(line_med_splited[0][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d'))
		last_record_date = max(datetime.strptime(line_med_splited[-1][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d'))
	elif (line_med_splited == [] and line_diag_splited != [] and line_proc_splited != []):
		first_record_date = min(datetime.strptime(line_proc_splited[0][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d'))
		last_record_date = max(datetime.strptime(line_proc_splited[-1][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d'))	
	elif (line_med_splited == [] and line_diag_splited != [] and line_proc_splited == []):
		# pdb.set_trace()
		first_record_date = datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d')
		last_record_date = datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d')
	else:
		pdb.set_trace()

	if current_patient_demog['MCI_label'].values[0] == 1:
		idx_date = datetime.strptime(current_patient_demog['index_date_OR_diag_date'].values[0][:10],'%Y-%m-%d')
	else:
		idx_date = last_record_date + relativedelta(months= -prediction_window_size)
	# pdb.set_trace()	
	for i in range(len(line_med_splited)):
		current_date = datetime.strptime(line_med_splited[i][0][:10], '%Y-%m-%d') 		
		current_date_to_idx_date = (idx_date-current_date).days
		# current_date_to_idx_date =  (idx_date.year - current_date.year) * 12 + idx_date.month - current_date.month 
		if current_date_to_idx_date >= (prediction_window_size*30):
			line_meds_blinded.append(line_med_splited[i])
	# pdb.set_trace()		
	for i in range(len(line_diag_splited)):
		current_date = datetime.strptime(line_diag_splited[i][0][:10], '%Y-%m-%d') 
		current_date_to_idx_date = (idx_date-current_date).days
		# current_date_to_idx_date =  (idx_date.year - current_date.year) * 12 + idx_date.month - current_date.month 
		if current_date_to_idx_date >= (prediction_window_size*30):
			line_diags_blinded.append(line_diag_splited[i])
	# pdb.set_trace()		
	for i in range(len(line_proc_splited)):
		current_date = datetime.strptime(line_proc_splited[i][0][:10], '%Y-%m-%d') 
		current_date_to_idx_date = (idx_date-current_date).days
		# current_date_to_idx_date =  (idx_date.year - current_date.year) * 12 + idx_date.month - current_date.month 
		if current_date_to_idx_date >= (prediction_window_size*30):
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
	if (line_med_splited != [] and line_diag_splited != [] and line_proc_splited != []):
		first_record_date = min(datetime.strptime(line_med_splited[0][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d'), datetime.strptime(line_proc_splited[0][0][:10], '%Y-%m-%d'))
		last_record_date = max(datetime.strptime(line_med_splited[-1][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d'), datetime.strptime(line_proc_splited[-1][0][:10], '%Y-%m-%d'))
	elif (line_med_splited != [] and line_diag_splited != [] and line_proc_splited == []):
		first_record_date = min(datetime.strptime(line_med_splited[0][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d'))
		last_record_date = max(datetime.strptime(line_med_splited[-1][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d'))
	elif (line_med_splited == [] and line_diag_splited != [] and line_proc_splited != []):
		first_record_date = min(datetime.strptime(line_proc_splited[0][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d'))
		last_record_date = max(datetime.strptime(line_proc_splited[-1][0], '%Y-%m-%d'), datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d'))	
	elif (line_med_splited == [] and line_diag_splited != [] and line_proc_splited == []):
		# pdb.set_trace()
		first_record_date = datetime.strptime(line_diag_splited[0][0][:10], '%Y-%m-%d')
		last_record_date = datetime.strptime(line_diag_splited[-1][0][:10], '%Y-%m-%d')
	else:
		pdb.set_trace()

	if current_patient_demog['MCI_label'].values[0] == 1:		
		diag_date = datetime.strptime(current_patient_demog['index_date_OR_diag_date'].values[0][:10],'%Y-%m-%d')
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
		elig_flag = False	
	
	return elig_flag



def create_longitudinal_ccs(patient_id, line_diag_splited, frequent_ccs_dict):
	# pdb.set_trace()
	# epsil = 2.220446049250313e-16
	# round_num_digit = 5
	longitudinal_codes = []
	for i in range(len(line_diag_splited)):
		longitudinal_codes.extend([line_diag_splited[i][0]])
		for j in range(1, len(line_diag_splited[i])): 
			if (line_diag_splited[i][j] in frequent_ccs_dict):           
				frequent_ccs_dict[ line_diag_splited[i][j]] = 1
			else:
				continue  
				print('test') 
		# pdb.set_trace()
		frequent_ccs_dict_sorted = dict(collections.OrderedDict(sorted(frequent_ccs_dict.items())))  
		longitudinal_codes.extend(frequent_ccs_dict_sorted.values())
		frequent_ccs_dict = dict.fromkeys(frequent_ccs_dict, 0) 
	# pdb.set_trace()	
	return longitudinal_codes

def create_longitudinal_med(line_med_splited, frequent_meds_dict):
	# pdb.set_trace()
	longitudinal_codes = []
	for i in range(len(line_med_splited)):
		longitudinal_codes.extend([line_med_splited[i][0]])
		for j in range(1, len(line_med_splited[i])): 
			if (line_med_splited[i][j] in frequent_meds_dict):           
				frequent_meds_dict[ line_med_splited[i][j]] = 1
			else:
				continue  
				print('test') 
		frequent_meds_dict_sorted = dict(collections.OrderedDict(sorted(frequent_meds_dict.items())))    
		longitudinal_codes.extend(frequent_meds_dict_sorted.values())
		frequent_meds_dict = dict.fromkeys(frequent_meds_dict, 0) 
	return longitudinal_codes

def create_longitudinal_proc(patient_id, line_proc_splited, frequent_procs_dict):
	longitudinal_codes = []
	for i in range(len(line_proc_splited)):
		longitudinal_codes.extend([line_proc_splited[i][0]])
		for j in range(1, len(line_proc_splited[i])): 
			if (line_proc_splited[i][j] in frequent_procs_dict):           
				frequent_procs_dict[ line_proc_splited[i][j]] = 1
			else:
				continue  
				print('test') 
		frequent_procs_dict_sorted = dict(collections.OrderedDict(sorted(frequent_procs_dict.items())))    
		longitudinal_codes.extend(frequent_procs_dict_sorted.values())
		frequent_procs_dict = dict.fromkeys(frequent_procs_dict, 0) 

	return longitudinal_codes

def sequence_sync(sequence_dias
					, sequence_procs
					, sequence_meds
					, top_n_ccs
					, top_n_proc
					, top_n_med
					, diag_header
					, proc_header
					, med_header
					, patient_id):
	# pdb.set_trace()
	patient_longitudinal_vec = []
	diags_reshaped = np.reshape(sequence_dias,(-1, top_n_ccs+1))
	diags_reshaped_pd = pd.DataFrame(data=diags_reshaped, columns=['timestamp']+ diag_header)
	if len(diags_reshaped_pd)>0:
		diags_reshaped_pd['timestamp'] = diags_reshaped_pd['timestamp'].str[:7].str.replace('-','')
		diags_reshaped_pd = diags_reshaped_pd.astype(int)

	proc_reshaped = np.reshape(sequence_procs,(-1, top_n_proc+1))
	proc_reshaped_pd = pd.DataFrame(data=proc_reshaped, columns=['timestamp']+ proc_header)
	if len(proc_reshaped_pd)>0:
		proc_reshaped_pd['timestamp'] = proc_reshaped_pd['timestamp'].str[:7].str.replace('-','')
		proc_reshaped_pd = proc_reshaped_pd.astype(int)

	meds_reshaped = np.reshape(sequence_meds,(-1, top_n_med+1))
	meds_reshaped_pd = pd.DataFrame(data=meds_reshaped, columns=['timestamp']+ med_header)
	if len(meds_reshaped_pd)>0:
		meds_reshaped_pd['timestamp'] = meds_reshaped_pd['timestamp'].str[:7].str.replace('-','')
		meds_reshaped_pd = meds_reshaped_pd.astype(int)
	
	all_times_str = diags_reshaped_pd.iloc[:,0].values.tolist() + proc_reshaped_pd.iloc[:,0].values.tolist() + meds_reshaped_pd.iloc[:,0].values.tolist()
	all_times_str = list(set(all_times_str))	
	all_times_str.sort()

	# pdb.set_trace()
	all_times_synched = vdates_syncher(all_times_str)
	seq_length = len(all_times_synched)
	for i in range(len(all_times_synched)):
		current_timestamp = all_times_synched[i]
		patient_longitudinal_vec.extend([current_timestamp])
		
		current_diags = diags_reshaped_pd[diags_reshaped_pd['timestamp'] == current_timestamp]
		if len(current_diags) >0:
			current_diags = current_diags.sum(axis=0)
			current_diags_list = current_diags.values.tolist()
			current_diags_list = [1 if x>0 else 0 for x in current_diags_list[1:] ]
			patient_longitudinal_vec.extend(current_diags_list)
		else:
			patient_longitudinal_vec.extend([0]*top_n_ccs)
				
		current_procs = proc_reshaped_pd[proc_reshaped_pd['timestamp'] == current_timestamp]
		if len(current_procs) > 0:
			current_procs = current_procs.sum(axis=0)
			current_procs_list = current_procs.values.tolist()
			current_procs_list = [1 if x>0 else 0 for x in current_procs_list[1:] ]
			patient_longitudinal_vec.extend(current_procs_list)
		else:
			patient_longitudinal_vec.extend([0]*top_n_proc)

		current_meds = meds_reshaped_pd[meds_reshaped_pd['timestamp'] == current_timestamp]
		if len(current_meds) > 0:
			current_meds = current_meds.sum(axis=0)
			current_meds_list = current_meds.values.tolist()
			current_meds_list = [1 if x>0 else 0 for x in current_meds_list[1:] ]
			patient_longitudinal_vec.extend(current_meds_list)
		else:
			patient_longitudinal_vec.extend([0]*top_n_med)

	return 	patient_longitudinal_vec, seq_length	


def create_longitudinal(diagnosis_file_path
					, medication_file_path
					, procedure_file_path
					, ccs_frequencies_mci_path
					, pharm_class_frequencies_mci_path
					, procedure_id_frequencies_mci_path
					, demographic_file_path
					, top_n_ccs							 
					, top_n_med
					, top_n_proc
					, prediction_window_size
					, cohort	
					, logging_milestone		
					, recommender
					, treatment_window
					, stationary_train_data_path
					, stationary_test_data_path
					):
	# pdb.set_trace()
	training_stationary_data = pd.read_csv(stationary_train_data_path)
	testing_stationary_data = pd.read_csv(stationary_test_data_path)
	target_col_names = [x for x in training_stationary_data.columns if 'target_proc_' in x]
	
	train_data_y = training_stationary_data[training_stationary_data.columns[training_stationary_data.columns.isin(['Patient_ID']+target_col_names)]]
	test_data_y = testing_stationary_data[testing_stationary_data.columns[testing_stationary_data.columns.isin(['Patient_ID']+target_col_names)]]

	max_seq_len = 252
	print('Reading demographic data ...')
	demographic_data = pd.read_csv(demographic_file_path)
	demographic_data['sex'] = demographic_data['sex'].map({'Male': 1, 'Female':2, 'Unknown': 0})
	demographic_data['canonical_race'] = demographic_data['canonical_race'].map({'Native American': 1, 'Black':2, 'White': 3, 'Pacific Islander':4, 'Asian':5, 'Unknown':6, 'Other':7})

	
	# Rread frequent ICD codes
	# pdb.set_trace()
	print('Check out dtype=str')
	print('Reading frequent codes and selecting top features')
	frequent_ccs = pd.read_csv(ccs_frequencies_mci_path)
	frequent_ccs.columns=frequent_ccs.columns.str.strip()
	frequent_ccs = frequent_ccs.sort_values('num patient', ascending=False)
	frequent_ccs_top_n = frequent_ccs['Code'].values[:top_n_ccs].tolist()


	frequent_meds = pd.read_csv(pharm_class_frequencies_mci_path)
	frequent_meds.columns=frequent_meds.columns.str.strip()
	frequent_meds = frequent_meds.sort_values('num patient', ascending=False)
	frequent_meds_top_n = frequent_meds['Code'].values[:top_n_med].tolist()

	frequent_procs = pd.read_csv(procedure_id_frequencies_mci_path)
	frequent_procs.columns = frequent_procs.columns.str.strip()
	frequent_procs = frequent_procs.sort_values('num patient', ascending=False)	
	frequent_procs_top_n = frequent_procs['Code'].values[:top_n_proc].tolist()

	# pdb.set_trace()
	frequent_ccs_dict = {}
	for i in range(len(frequent_ccs_top_n)):
		frequent_ccs_dict[frequent_ccs_top_n[i]] = 0 
	frequent_ccs_dict = dict(collections.OrderedDict(sorted(frequent_ccs_dict.items())))    

	frequent_meds_dict = {}
	for i in range(len(frequent_meds_top_n)):
		frequent_meds_dict[str(frequent_meds_top_n[i])] = 0 
	frequent_meds_dict = dict(collections.OrderedDict(sorted(frequent_meds_dict.items())))    

	frequent_procs_dict = {}
	for i in range(len(frequent_procs_top_n)):
		frequent_procs_dict[str(frequent_procs_top_n[i])] = 0 
	frequent_procs_dict = dict(collections.OrderedDict(sorted(frequent_procs_dict.items())))    


	# pdb.set_trace()
	with open(diagnosis_file_path) as diag_file, open(medication_file_path) as med_file, open(procedure_file_path) as proc_file, open('longitudinal_data/'+recommender+'longitudinal_train.csv', 'w') as long_train_file, open('longitudinal_data/'+recommender+'longitudinal_test.csv', 'w') as long_test_file:
		header_diag = next(diag_file)
		header_med = next(med_file)
		header_proc = next(proc_file)

		# elig_patients_file.write('eligible_patients_ids\n')
		# unelig_patients_file.write('uneligible_patients_ids\n')

		long_train_file.write('Patient_ID, timestamp, '+ (','.join([*frequent_ccs_dict.keys()])) )
		long_train_file.write(',')

		long_train_file.write(','.join(str(x) for x in [*frequent_procs_dict.keys()]))
		long_train_file.write(',')

		long_train_file.write(','.join(str(x) for x in [*frequent_meds_dict.keys()]))
		long_train_file.write(',')  

		long_train_file.write(','.join(target_col_names))
		long_train_file.write(',') 

		long_train_file.write('sequence_length')    
		long_train_file.write('\n')       


		long_test_file.write('Patient_ID, timestamp, '+ (','.join([*frequent_ccs_dict.keys()])) )
		long_test_file.write(',')

		long_test_file.write(','.join(str(x) for x in [*frequent_procs_dict.keys()]))
		long_test_file.write(',')

		long_test_file.write(','.join(str(x) for x in [*frequent_meds_dict.keys()]))
		long_test_file.write(',')  

		long_test_file.write(','.join(target_col_names))
		long_test_file.write(',') 

		long_test_file.write('sequence_length')    
		long_test_file.write('\n')       		

		line_counter = 0
		
		# pdb.set_trace()
		
		previous_id = '0'
		read_med = True
		read_proc = True

		# num_lines_in_diag = sum(1 for line in diag_file)
		for line in diag_file:
			line_counter+=1
			if line_counter %1000 ==0:
				print('Processing line {} in the diagnosis table.'.format(line_counter))
			# print(line_counter)
			line_diag = line.replace('\n','').split(',')
			line_diag_splited = [list(y) for x, y in itertools.groupby(line_diag[1:], lambda z: z == 'EOV') if not x]    
	
			# line_med = meds_data[ meds_data[0] == line_diag[0]]
			if read_med == True:
				line_med = med_file.readline().replace('\n','').split(',')
				line_med_splited = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]    

			if read_proc == True:	
				line_proc = proc_file.readline().replace('\n','').split(',')
				line_proc_splited = [list(y) for x, y in itertools.groupby(line_proc[1:], lambda z: z == 'EOV') if not x]    
			
			current_id = line_diag[0]
			if previous_id>current_id:
				pdb.set_trace()
				print('Previous ID larger')
			previous_id = current_id
			
			# logging.info('ID in diagnosis record is {}. Id in medication record is {}. ID in procedure record is {}.'.format(line_diag[0], line_med[0], line_proc[0]))

			if not (line_diag[0] == line_med[0] == line_proc[0]):
				# pdb.set_trace()
				logging.info('========================================================')
				logging.info('Patients dont match')
				logging.info('========================================================')

			if line_diag[0] == line_med[0]:
				logging.info('line_diag[0] == line_med[0]: {} and {}. therefore read_med=True.'.format(line_diag[0], line_med[0]))
				line_med_splited = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]    
				read_med = True
			elif line_diag[0] < line_med[0]:
				# pdb.set_trace()
				logging.info('line_diag[0] < line_med[0]: {} and {}'.format(line_diag[0], line_med[0]))
				line_med_splited = []
				logging.info('Setting line_med_splited=[] and read_med = False')
				read_med = False
			elif line_diag[0] > line_med[0]:
				logging.info('line_diag[0] > line_med[0]: {} and {}'.format(line_diag[0], line_med[0]))
				logging.info('start reading lines from med while line_med[0] < line_diag[0]')
				while line_med[0] < line_diag[0]:
					line_med = med_file.readline().replace('\n','').split(',')
					line_med_splited = [list(y) for x, y in itertools.groupby(line_med[1:], lambda z: z == 'EOV') if not x]    
					logging.info('Read line_med[0]: {}'.format(line_med[0]))
					if line_med == ['']:
						line_med_splited = []
						# pdb.set_trace()
						break					
				# pdb.set_trace()
				if line_med[0] == line_diag[0]:
					logging.info('line_med[0] == line_diag[0] and therefore read_med = True')
					read_med = True
				else:
					logging.info('line_med[0] != line_diag[0] and therefore read_med = False')
					read_med = False	

			if line_diag[0] == line_proc[0]:
				logging.info('line_diag[0] == line_proc[0]: {} and {}. therefore read_proc=True.'.format(line_diag[0], line_proc[0]))				
				line_proc_splited = [list(y) for x, y in itertools.groupby(line_proc[1:], lambda z: z == 'EOV') if not x]    
				read_proc = True
			elif line_diag[0] < line_proc[0]:
				logging.info('line_diag[0] < line_proc[0]: {} and {}'.format(line_diag[0], line_proc[0]))
				line_proc_splited = []
				logging.info('Setting line_proc_splited=[] and read_proc = False')
				read_proc = False
			elif line_diag[0] > line_proc[0]:
				logging.info('line_diag[0] > line_proc[0]: {} and {}'.format(line_diag[0], line_proc[0]))
				logging.info('start reading lines from proc while line_proc[0] < line_diag[0]')
				while line_proc[0] < line_diag[0]:
					line_proc = proc_file.readline().replace('\n','').split(',')
					line_proc_splited = [list(y) for x, y in itertools.groupby(line_proc[1:], lambda z: z == 'EOV') if not x]    
					logging.info('Read line_proc[0]: {}'.format(line_proc[0]))
					if line_proc == ['']:
						line_proc_splited = []
						# pdb.set_trace()
						break					
				# pdb.set_trace()
				if line_proc[0] == line_diag[0]:
					logging.info('line_proc[0] == line_diag[0] and therefore read_proc = True')
					read_proc = True
				else:
					logging.info('line_med[0] != line_proc[0] and therefore read_proc = False')
					read_proc = False	


			current_patient_demog = demographic_data[demographic_data['anon_id']==line_med[0]]

			# pdb.set_trace()
			current_in_train = train_data_y[train_data_y['Patient_ID'] == line_diag[0]]
			current_in_test = test_data_y[test_data_y['Patient_ID'] == line_diag[0]]

			if len(current_in_train) == 1 and len(current_in_test) == 0:
				current_patient_target = current_in_train.drop(['Patient_ID'], axis=1).values.tolist()[0]
			elif len(current_in_train) == 0 and len(current_in_test) == 1:
				current_patient_target = current_in_test.drop(['Patient_ID'], axis=1).values.tolist()[0]
			elif len(current_in_train) == 0 and len(current_in_test) == 0:
				continue
			else:
				print('Warning: patient is in both train and test')
				pdb.set_trace()	

			if current_patient_demog.empty:
				print('Demographic data is missing ...')
				continue
			# pdb.set_trace()

			if len(line_diag_splited)==0:
				diag_id = 'NA'
			else:
				diag_id = line_diag[0]

			if len(line_med_splited)==0:
				med_id = 'NA'
			else:
				med_id = line_med[0]

			if len(line_proc_splited)==0:
				proc_id = 'NA'
			else:
				proc_id = line_proc[0]
			
			logging.info('line_diag[0], line_med[0], line_proc[0]: {}, {}, {}'.format(diag_id, med_id, proc_id))

			# pdb.set_trace()
			# elig_flag = check_data_availibility(line_med_splited
			# 										, line_diag_splited
			# 										, line_proc_splited
			# 										, current_patient_demog
			# 										, prediction_window_size)

			# if elig_flag == True:
			# 	elig_patients_file.write(line_med[0])
			# 	elig_patients_file.write('\n')
			# else:	
			# 	unelig_patients_file.write(line_med[0])
			# 	unelig_patients_file.write('\n')
			# 	continue
			
			# pdb.set_trace()
			line_med_blinded, line_diag_blinded, line_proc_blinded= blind_data(line_med[0]
																				,line_med_splited
																				, line_diag_splited
																				, line_proc_splited
																				, current_patient_demog
																				, prediction_window_size)


			frequent_ccs_dict = dict.fromkeys(frequent_ccs_dict, 0)    
			frequent_meds_dict = dict.fromkeys(frequent_meds_dict, 0)    
			frequent_procs_dict = dict.fromkeys(frequent_procs_dict, 0)    
			

			longitudinal_ccs_vect = create_longitudinal_ccs(line_med[0], line_diag_blinded, frequent_ccs_dict)		
			longitudinal_med_vect = create_longitudinal_med(line_med_blinded, frequent_meds_dict)					
			longitudinal_proc_vect = create_longitudinal_proc(line_med[0], line_proc_blinded, frequent_procs_dict)		
			
			# if line_diag[0] == 'JC2a21670':
			# 	pdb.set_trace()
			current_patient_sequence, current_patient_sequence_length = sequence_sync(longitudinal_ccs_vect
										, longitudinal_proc_vect
										, longitudinal_med_vect
										, top_n_ccs
										, top_n_proc
										, top_n_med
										, [*frequent_ccs_dict.keys()]
										, [*frequent_procs_dict.keys()]
										, [*frequent_meds_dict.keys()]
										, line_diag[0])
			if current_patient_sequence_length > max_seq_len:
				pdb.set_trace()
			zero_padding_length = (max_seq_len-current_patient_sequence_length) * (top_n_med + top_n_proc + top_n_ccs + 1)
			current_patient_sequence.extend([0]*zero_padding_length)

			# current_patient_age = int(current_patient_demog['index_date_OR_diag_date'].values[0].split('-')[0])-int(current_patient_demog['bdate'].values[0].split('-')[0])
			# if current_patient_age<50:
			# 	continue
			# pdb.set_trace()
			if len(current_in_train) == 1 and len(current_in_test) == 0:
				long_train_file.write((line_diag[0]))
				long_train_file.write(',')				
				long_train_file.write(','.join(map(repr, list(current_patient_sequence))))
				long_train_file.write(',')
				long_train_file.write(','.join(map(repr, list(current_patient_target))))
				long_train_file.write(',')				
				long_train_file.write(str(current_patient_sequence_length))
				long_train_file.write('\n')            				
			elif len(current_in_train) == 0 and len(current_in_test) == 1:
				long_test_file.write((line_diag[0]))
				long_test_file.write(',')				
				long_test_file.write(','.join(map(repr, list(current_patient_sequence))))
				long_test_file.write(',')
				long_test_file.write(','.join(map(repr, list(current_patient_target))))
				long_test_file.write(',')				
				long_test_file.write(str(current_patient_sequence_length))
				long_test_file.write('\n')            

	logging.info('Finished creating stationary data for the cohort {}'.format(cohort))
	logging.info('***********************************************************************')			
	logging.info('***********************************************************************')			
	logging.info('***********************************************************************')			
	logging.info('***********************************************************************')			
	logging.info('***********************************************************************')							
