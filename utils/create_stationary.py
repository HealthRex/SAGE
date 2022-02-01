import pdb
import pandas as pd
import itertools
import math
import collections
import numpy as np

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
					, top_n
					):
	
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
	with open(diagnosis_file_path) as diag_file, open(medication_file_path) as med_file, open(procedure_file_path) as proc_file, open('stationary_data/stationary_dataset.csv', 'w') as stationary_file:
		header_diag = next(diag_file)
		header_med = next(med_file)
		header_proc = next(proc_file)

		stationary_file.write('Patient_ID, '+ (','.join([*frequent_icd10s_dict.keys()])) )
		stationary_file.write(',')

		stationary_file.write(','.join([*frequent_icd9s_dict.keys()]))
		stationary_file.write(',')       

		stationary_file.write(','.join(str(x) for x in [*frequent_meds_dict.keys()]))
		stationary_file.write(',')  

		stationary_file.write(','.join(str(x) for x in [*frequent_procs_dict.keys()]))
		stationary_file.write('\n')          

		for line in diag_file:
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
			
			frequent_icd10s_dict = dict.fromkeys(frequent_icd10s_dict, 0)    
			frequent_icd9s_dict = dict.fromkeys(frequent_icd9s_dict, 0)    
			frequent_meds_dict = dict.fromkeys(frequent_meds_dict, 0)    
			frequent_procs_dict = dict.fromkeys(frequent_procs_dict, 0)    
			
			stationary_icd10_vect = create_stationary_icd10(line_diag_splited, frequent_icd10s_dict)		
			stationary_icd9_vect = create_stationary_icd9(line_diag_splited, frequent_icd9s_dict)		
			stationary_med_vect = create_stationary_med(line_med_splited, frequent_meds_dict)					
			stationary_proc_vect = create_stationary_proc(line_proc_splited, frequent_procs_dict)		
            
			stationary_file.write((line_med[0]))
			stationary_file.write(',')
			stationary_file.write(','.join(map(repr, list(stationary_icd10_vect.values()))))
			stationary_file.write(',')
			stationary_file.write(','.join(map(repr, list(stationary_icd9_vect.values()))))
			stationary_file.write(',')
			stationary_file.write(','.join(map(repr, list(stationary_med_vect.values()))))
			stationary_file.write(',')
			stationary_file.write(','.join(map(repr, list(stationary_proc_vect.values()))))
			# stationary_file.write(',')            
			# stationary_file.write(str(current_patient_age))
			# stationary_file.write(',')
			# stationary_file.write(str(current_patient_sex))
			# stationary_file.write(',')
			# stationary_file.write(str(line_demog[label_idx]))
			stationary_file.write('\n')            

			print('End')
 #    for i in range(len(line_med_splitted)):
 #        # if line_med_splitted[i][0]
 #        for j in range(1, len(line_med_splitted[i])): 
 #            if (line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]+'_tcgp_2digit') in distinct_tcgpid_2digit_dict:           
 #                distinct_tcgpid_2digit_dict[ line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits]+'_tcgp_2digit'] += 1
 #            elif line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits] == 'NO' or line_med_splitted[i][j].replace("'","")[:tcgpi_num_digits] =='EO':
 #                continue
 #            else:
 #                pdb.set_trace()   
 #                print('test') 
 #    distinct_tcgpid_2digit_dict_sorted = dict(collections.OrderedDict(sorted(distinct_tcgpid_2digit_dict.items())))    
 #    num_records = len(line_med_splitted)
 #    if (num_records > 1):
 #        distinct_tcgpid_2digit_dict_sorted = {k: np.round(v / (math.log(num_records, 2)+ epsil), round_num_digit) for k, v in distinct_tcgpid_2digit_dict_sorted.items()}
 #    else:
 #        distinct_tcgpid_2digit_dict_sorted = {k: np.round(v , round_num_digit) for k, v in distinct_tcgpid_2digit_dict_sorted.items()}    

 #    return distinct_tcgpid_2digit_dict_sorted

					
	# 		print('do something')
	# print('The end')
# def reformat_medication():

# def reformat_procedure():