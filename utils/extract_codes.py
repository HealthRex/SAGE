from google.cloud import bigquery
from google.cloud.bigquery import dbapi
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import *
import pdb
import os
import time
import logging
import csv

def extract_diagnosis(client_name
                    , query_diag
                    , icd_to_ccs_table_query
                    , icd9_to_icd10_query
                    , cohort
                    , icd10_field_name
                    , icd9_field_name   
                    , diag_time_field_name  
                    , diag_patient_id_field_name               
                    , display_step):
    # pdb.set_trace()
    # end_of_diag_file = False
    # execute diagnosis query
    proc_start_time = time.time()
    client = bigquery.Client(client_name); 
    conn = dbapi.connect(client);
    cursor = conn.cursor();
    cursor.execute(query_diag);    
    results = cursor.fetchall();
    num_fields = len(cursor.description)
    field_names = [i[0] for i in cursor.description]

    # ==== Read icd to ccs map
    icd_to_ccs_table = pd.read_sql_query(icd_to_ccs_table_query, conn); 
    icd_to_ccs_table['ICD10'] = icd_to_ccs_table['ICD10'].str.strip()
    icd_to_ccs_table['ICD10_string'] = icd_to_ccs_table['ICD10_string'].str.strip()
    icd_to_ccs_table['CCSR_CATEGORY_1'] = icd_to_ccs_table['CCSR_CATEGORY_1'].str.strip()

    # ==== Read icd9 to icd10 table
    icd9_to_icd10 = pd.read_sql_query(icd9_to_icd10_query, conn); 
    icd9_to_icd10['icd9_string'] = icd9_to_icd10['icd9_string'].str.strip()
    icd9_to_icd10['icd10_string'] = icd9_to_icd10['icd10_string'].str.strip()

    # ==== Reading unique CCS codes
    unique_ccs_codes = icd_to_ccs_table['CCSR_CATEGORY_1'].unique()
    unique_ccs_dict = {}
    for i in range(len(unique_ccs_codes)):
        if unique_ccs_codes[i] not in unique_ccs_dict:
            unique_ccs_dict[unique_ccs_codes[i]] = 0

    with open('intermediate_files/diagnosis_codes_'+cohort+'.csv', 'w') as diag_file:
        diag_file.write('patient id, timestamp, icd10 and icd9 codes, end of visit token\n')
        #==== While not end of the diagnoses file
        line_counter = 0
        patient_num = 0
        total_num_records = len(results)

        line_diag = results[line_counter]
        while line_counter < total_num_records:
            if line_counter%display_step==0:
                print('Finished processing {} diagnoses records out of {} records.'.format(line_counter,total_num_records))
            #==== Reading diagnoses visits for current patients           
            current_id_diag = line_diag[diag_patient_id_field_name]
            current_patient_diags = []

            while current_id_diag == line_diag[diag_patient_id_field_name]:
                current_patient_diags.append(line_diag)
                line_counter+=1
                if line_counter >= total_num_records:
                    break

                line_diag = results[line_counter]            
                if line_counter%display_step==0:
                    print('Finished processing {} diagnoses records out of {} records.'.format(line_counter,total_num_records))                

            # pdb.set_trace()
            # print(line_counter)
            current_patient_diags_pd = pd.DataFrame(np.array(current_patient_diags), columns= field_names)
            diagnosis_data_grouped = current_patient_diags_pd.groupby(by=diag_time_field_name)
            # current_patient_diags_ar = np.array(current_patient_diags, dtype='U')       
            diag_file.write(current_id_diag)
            
            # if current_id_diag == 'JCcb67b1':
            # pdb.set_trace()
            current_patient_all_ccs = []
            for timestamp, group in diagnosis_data_grouped:
                diag_file.write(',')
                diag_file.write(str(timestamp))                
                diag_file.write(',')
                
                current_icd10s = [x for x in group.loc[group[icd10_field_name].notna()][icd10_field_name].values.tolist()]                
                current_icd10s = [x.strip() for x in current_icd10s]
                current_icd10s_strings = [x.replace('.','') for x in current_icd10s]
                current_icd_to_ccs = icd_to_ccs_table[icd_to_ccs_table['ICD10'].isin(current_icd10s) | icd_to_ccs_table['ICD10_string'].isin(current_icd10s_strings)]
                current_ccs_codes_icd10 = current_icd_to_ccs['CCSR_CATEGORY_1'].unique().tolist()
                current_ccs_codes_icd10 = list(set(current_ccs_codes_icd10))

                current_icd9s = [x for x in group.loc[group[icd9_field_name].notna() & group[icd10_field_name].isna()][icd9_field_name].values.tolist()]    
                current_icd9s = [x.strip() for x in current_icd9s]
                current_icd9s_strings = [x.replace('.','') for x in current_icd9s]
                current_icd9_to_icd10 = icd9_to_icd10[icd9_to_icd10['icd9_string'].isin(current_icd9s_strings)]
                current_icd9_ccs = icd_to_ccs_table[icd_to_ccs_table['ICD10_string'].isin(current_icd9_to_icd10['icd10_string'].values.tolist()) ]
                current_ccs_codes_icd9 = current_icd9_ccs['CCSR_CATEGORY_1'].unique().tolist()
                current_ccs_codes_icd9 = list(set(current_ccs_codes_icd9))
                
                current_ccs_codes_all = current_ccs_codes_icd9 + current_ccs_codes_icd10
                current_ccs_codes_all = list(set(current_ccs_codes_all))
                if current_ccs_codes_all:   
                    # pdb.set_trace()                   
                    diag_file.write(','.join(current_ccs_codes_all))
                    diag_file.write(',')
                    current_patient_all_ccs.extend(current_ccs_codes_all)

                diag_file.write('EOV')
            # pdb.set_trace()
            diag_file.write('\n')
            
            # Computing stats for ICD codes
            current_patient_all_ccs = list(set(current_patient_all_ccs))
            for icd_idx in range(len(current_patient_all_ccs)):
                if current_patient_all_ccs[icd_idx] in unique_ccs_dict:
                    unique_ccs_dict[current_patient_all_ccs[icd_idx]] +=1
                else:
                    pdb.set_trace()
                    print('ICD does not exists')
            # pdb.set_trace()
            patient_num += 1
            # if patient_num%logging_milestone==0:
            #     logging.info('Completed extracting and writing the diagnoses stream for {} patient with ENROLID = {}.'.format(cohort, current_id_diag))
    # pdb.set_trace()
    print('Finished processing all {} of diagnoses records for the cohort in {}'.format(total_num_records, (time.time() - proc_start_time)))            
    with open('intermediate_files/ccs_frequencies_'+cohort+'.csv', 'w') as csv_file:  
        csv_file.write('Code, num patient\n')
        writer = csv.writer(csv_file)
        for key, value in unique_ccs_dict.items():
           writer.writerow([key, value])    
       
    # logging.info('Finished processing all {} of diagnoses records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
    # logging.info('================================')    
    # return 0      


#========= Medications================
def extract_medication(client_name
                    , query_med
                    , unique_medication_id_query
                    , cohort
                    , med_code_field_name
                    , med_time_field_name
                    , med_patient_id_field_name                      
                    , display_step):
    # pdb.set_trace()
    # end_of_med_file = False
    # execute medication query
    proc_start_time = time.time()
    client = bigquery.Client(client_name); 
    conn = dbapi.connect(client);
    cursor = conn.cursor();
    cursor.execute(query_med);    
    results = cursor.fetchall();
    num_fields = len(cursor.description)
    field_names = [i[0] for i in cursor.description]



    # ==== Reading unique medication codes

    cursor.execute(unique_medication_id_query);    
    results_unique_pharm_class = cursor.fetchall();
    unique_pharm_class = [item[0] for item in results_unique_pharm_class]
    unique_pharm_class_dict = {}
    for i in range(len(unique_pharm_class)):
        if unique_pharm_class[i] not in unique_pharm_class_dict:
            unique_pharm_class_dict[unique_pharm_class[i]] = 0


    with open('intermediate_files/medication_codes_'+cohort+'.csv', 'w') as med_file:
        med_file.write('patient id, order_time_jittered, pharm_class, end of visit token\n')
        #==== While not end of the diagnoses file
        line_counter = 0
        patient_num = 0
        total_num_records = len(results)

        line_med = results[line_counter]
        while line_counter < total_num_records:
            if line_counter%display_step==0:
                print('Finished processing {} medication records out of {} records.'.format(line_counter,total_num_records))
            #==== Reading diagnoses visits for current patients           
            current_id_med = line_med[med_patient_id_field_name]
            current_patient_med = []

            while current_id_med == line_med[med_patient_id_field_name]:
                current_patient_med.append(line_med)
                line_counter+=1
                if line_counter >= total_num_records:
                    break

                line_med = results[line_counter]            
                if line_counter%display_step==0:
                    print('Finished processing {} diagnoses records out of {} records.'.format(line_counter,total_num_records))                

            # pdb.set_trace()
            # print(line_counter)
            current_patient_med_pd = pd.DataFrame(np.array(current_patient_med), columns= field_names)
            medication_data_grouped = current_patient_med_pd.groupby(by=med_time_field_name)
            # current_patient_med_ar = np.array(current_patient_med, dtype='U')       
            med_file.write(current_id_med)
            
            # pdb.set_trace()
            current_patient_all_pharm_class = []
            for timestamp, group in medication_data_grouped:
                med_file.write(',')
                med_file.write(str(timestamp))                
                med_file.write(',')
                
                current_med_ids = group.loc[group[med_code_field_name].notna()][med_code_field_name].values.tolist()
                current_med_ids = list(set(current_med_ids))
                if current_med_ids:
                    med_file.write(','.join([str(x) for x in current_med_ids]))
                    med_file.write(',')
                    current_patient_all_pharm_class.extend(current_med_ids)
                med_file.write('EOV')
            med_file.write('\n')
            # pdb.set_trace()
            # Computing stats for medication codes
            current_patient_all_pharm_class = list(set(current_patient_all_pharm_class))
            for pharm_class_idx in range(len(current_patient_all_pharm_class)):
                if current_patient_all_pharm_class[pharm_class_idx] in unique_pharm_class_dict:
                    unique_pharm_class_dict[current_patient_all_pharm_class[pharm_class_idx]] +=1
                else:
                    pdb.set_trace()
                    print('ICD does not exists')
            
            patient_num += 1
            # if patient_num%logging_milestone==0:
            #     logging.info('Completed extracting and writing the diagnoses stream for {} patient with ENROLID = {}.'.format(cohort, current_id_diag))
    print('Finished processing all {} of medication records for the cohort in {}'.format(total_num_records, (time.time() - proc_start_time)))            
    with open('intermediate_files/pharm_class_frequencies_'+cohort+'.csv', 'w') as csv_file:  
        csv_file.write('Code, num patient\n')
        writer = csv.writer(csv_file)
        for key, value in unique_pharm_class_dict.items():
           writer.writerow([key, value])    
        # logging.info('Finished processing all {} of diagnoses records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
        # logging.info('================================')    
    # return 0      



#========= procedures================
def extract_procedure(client_name
                    , query_proc
                    , unique_proc_id_query
                    , cohort
                    , proc_code_field_name
                    , proc_time_field_name
                    , proc_patient_id_field_name                     
                    , display_step):
    # pdb.set_trace()
    # end_of_proc_file = False
    # execute medication query
    proc_start_time = time.time()
    client = bigquery.Client(client_name); 
    conn = dbapi.connect(client);
    cursor = conn.cursor();
    cursor.execute(query_proc);    
    results = cursor.fetchall();
    num_fields = len(cursor.description)
    field_names = [i[0] for i in cursor.description]

    # ==== Reading unique procedure codes
    cursor.execute(unique_proc_id_query);    
    results_unique_proc_id = cursor.fetchall();
    unique_proc_id = [item[0] for item in results_unique_proc_id]
    unique_proc_id_dict = {}
    for i in range(len(unique_proc_id)):
        if unique_proc_id[i] not in unique_proc_id_dict:
            unique_proc_id_dict[unique_proc_id[i]] = 0


    with open('intermediate_files/procedure_codes_'+cohort+'.csv', 'w') as proc_file:
        proc_file.write('patient id, order_time_jittered_utc, proc_id, end of visit token\n')
        #==== While not end of the diagnoses file
        line_counter = 0
        patient_num = 0
        total_num_records = len(results)

        line_proc = results[line_counter]
        while line_counter < total_num_records:
            if line_counter%display_step==0:
                print('Finished processing {} procedure records out of {} records.'.format(line_counter,total_num_records))
            #==== Reading diagnoses visits for current patients           
            current_id_proc = line_proc[proc_patient_id_field_name]
            current_patient_proc = []
            # pdb.set_trace()
            while current_id_proc == line_proc[proc_patient_id_field_name]:
                current_patient_proc.append(line_proc)
                line_counter+=1
                if line_counter >= total_num_records:
                    break

                line_proc = results[line_counter]            
                if line_counter%display_step==0:
                    print('Finished processing {} diagnoses records out of {} records.'.format(line_counter,total_num_records))                

            # if current_id_proc== 'JC29fcdc5':
            #     pdb.set_trace()
            # print(line_counter)
            current_patient_proc_pd = pd.DataFrame(np.array(current_patient_proc), columns= field_names)
            procedure_data_grouped = current_patient_proc_pd.groupby(by=proc_time_field_name)
            # current_patient_proc_ar = np.array(current_patient_proc, dtype='U')       
            proc_file.write(current_id_proc)
            
            # pdb.set_trace()
            current_patient_all_proc_id = []           
            for timestamp, group in procedure_data_grouped:
                proc_file.write(',')
                proc_file.write(str(timestamp))                
                proc_file.write(',')
                
                current_proc_ids = group.loc[group[proc_code_field_name].notna()][proc_code_field_name].values.tolist()
                current_proc_ids = list(set(current_proc_ids))
                if current_proc_ids:
                    proc_file.write(','.join([str(x) for x in current_proc_ids]))
                    proc_file.write(',')
                    current_patient_all_proc_id.extend(current_proc_ids)

                proc_file.write('EOV')
            # pdb.set_trace()
            proc_file.write('\n')
            patient_num += 1
            # Computing stats for medication codes
            current_patient_all_proc_id = list(set(current_patient_all_proc_id))
            for proc_id_idx in range(len(current_patient_all_proc_id)):
                if current_patient_all_proc_id[proc_id_idx] in unique_proc_id_dict:
                    unique_proc_id_dict[current_patient_all_proc_id[proc_id_idx]] +=1
                else:
                    pdb.set_trace()
                    print('ICD does not exists')            
            # if patient_num%logging_milestone==0:
            #     logging.info('Completed extracting and writing the diagnoses stream for {} patient with ENROLID = {}.'.format(cohort, current_id_diag))
    print('Finished processing all {} of procedure records for the cohort in {}'.format(total_num_records, (time.time() - proc_start_time)))            
    with open('intermediate_files/procedure_id_frequencies_'+cohort+'.csv', 'w') as csv_file:  
        csv_file.write('Code, num patient\n')
        writer = csv.writer(csv_file)
        for key, value in unique_proc_id_dict.items():
           writer.writerow([key, value])    
        
        # logging.info('Finished processing all {} of diagnoses records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
        # logging.info('================================')    
    # return 0      




