import pdb
import pandas as pd
import itertools

def create_stationary_diagnosis():
	pdb.set_trace()

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


	pdb.set_trace()
	with open(diagnosis_file_path) as diag_file, open(medication_file_path) as med_file, open(procedure_file_path) as proc_file:
		header = next(diag_file)
		for line in diag_file:
			line_diag = line.replace('\n','').split(',')
			line_diag = [list(y) for x, y in itertools.groupby(line_diag[1:], lambda z: z == 'EOV') if not x]    
	
            line_med = med_file.readline()
            line_med = med_file.split(',')

            line_proc = proc_file.readline()
            line_proc = proc_file.split(',')
			
            # line_demog = demogs_filename.readline().rstrip('\n')
            # line_demog = line_demog.split(',')  

			create_stationary_diagnosis()		

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
 #        distinct_tcgpid_2digit_dict_sorted = {k: np.round(v / (math.log(num_records, 2)+ epsil), round_dig) for k, v in distinct_tcgpid_2digit_dict_sorted.items()}
 #    else:
 #        distinct_tcgpid_2digit_dict_sorted = {k: np.round(v , round_dig) for k, v in distinct_tcgpid_2digit_dict_sorted.items()}    

 #    return distinct_tcgpid_2digit_dict_sorted

        			
	# 		print('do something')
	# print('The end')
# def reformat_medication():

# def reformat_procedure():