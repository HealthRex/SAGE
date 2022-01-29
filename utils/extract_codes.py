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

def extract_diagnosis(client_name
                    , query_diag
                    , cohort
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
    # while 

    # for row in results:
    #     print( row );    

    # diagnosis_data = pd.read_sql_query(query_diag, conn);        




    #=== Reading first line of diagnoses 

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
            current_id_diag = line_diag['jc_uid']
            current_patient_diags = []

            while current_id_diag == line_diag['jc_uid']:
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
            diagnosis_data_grouped = current_patient_diags_pd.groupby(by='timestamp_utc')
            # current_patient_diags_ar = np.array(current_patient_diags, dtype='U')       
            diag_file.write(current_id_diag)
            diag_file.write(',')
            # if current_id_diag == 'JCcb67b1':
            #     pdb.set_trace()
            for timestamp, group in diagnosis_data_grouped:
                diag_file.write(str(timestamp))                
                diag_file.write(',')
                
                current_icd10s = ['ICD10_'+x for x in group.loc[group['icd10'].notna()]['icd10'].values.tolist()]
                if current_icd10s:
                    diag_file.write(','.join(current_icd10s))
                    diag_file.write(',')

                current_icd9s = ['ICD9_'+x for x in group.loc[group['icd9'].notna()]['icd9'].values.tolist()]    
                if current_icd9s:                      
                    diag_file.write(','.join(current_icd9s))
                    diag_file.write(',')

                diag_file.write('EOV,')
            
            diag_file.write('\n')

            patient_num += 1
            # if patient_num%logging_milestone==0:
            #     logging.info('Completed extracting and writing the diagnoses stream for {} patient with ENROLID = {}.'.format(cohort, current_id_diag))
        print('Finished processing all {} of diagnoses records for the cohort in {}'.format(total_num_records, (time.time() - proc_start_time)))            
        # logging.info('Finished processing all {} of diagnoses records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
        # logging.info('================================')    
    # return 0      


#========= Medications================
def extract_medication(client_name
                    , query_med
                    , cohort
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


    with open('intermediate_files/medication_codes_'+cohort+'.csv', 'w') as med_file:
        med_file.write('patient id, order_time_jittered, medication_id, end of visit token\n')
        #==== While not end of the diagnoses file
        line_counter = 0
        patient_num = 0
        total_num_records = len(results)

        line_med = results[line_counter]
        while line_counter < total_num_records:
            if line_counter%display_step==0:
                print('Finished processing {} medication records out of {} records.'.format(line_counter,total_num_records))
            #==== Reading diagnoses visits for current patients           
            current_id_med = line_med['jc_uid']
            current_patient_med = []

            while current_id_med == line_med['jc_uid']:
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
            medication_data_grouped = current_patient_med_pd.groupby(by='order_time_jittered_utc')
            # current_patient_med_ar = np.array(current_patient_med, dtype='U')       
            med_file.write(current_id_med)
            med_file.write(',')
            # pdb.set_trace()
            for timestamp, group in medication_data_grouped:
                med_file.write(str(timestamp))                
                med_file.write(',')
                
                current_med_ids = group.loc[group['medication_id'].notna()]['medication_id'].values.tolist()
                if current_med_ids:
                    med_file.write(','.join([str(x) for x in current_med_ids]))
                    med_file.write(',')
                med_file.write('EOV,')
            med_file.write('\n')
            patient_num += 1
            # if patient_num%logging_milestone==0:
            #     logging.info('Completed extracting and writing the diagnoses stream for {} patient with ENROLID = {}.'.format(cohort, current_id_diag))
        print('Finished processing all {} of medication records for the cohort in {}'.format(total_num_records, (time.time() - proc_start_time)))            
        # logging.info('Finished processing all {} of diagnoses records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
        # logging.info('================================')    
    # return 0      



#========= procedures================
def extract_procedure(client_name
                    , query_proc
                    , cohort
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
            current_id_proc = line_proc['jc_uid']
            current_patient_proc = []
            # pdb.set_trace()
            while current_id_proc == line_proc['jc_uid']:
                current_patient_proc.append(line_proc)
                line_counter+=1
                if line_counter >= total_num_records:
                    break

                line_proc = results[line_counter]            
                if line_counter%display_step==0:
                    print('Finished processing {} diagnoses records out of {} records.'.format(line_counter,total_num_records))                

            # pdb.set_trace()
            # print(line_counter)
            current_patient_med_pd = pd.DataFrame(np.array(current_patient_proc), columns= field_names)
            procedure_data_grouped = current_patient_med_pd.groupby(by='order_time_jittered_utc')
            # current_patient_proc_ar = np.array(current_patient_proc, dtype='U')       
            proc_file.write(current_id_proc)
            proc_file.write(',')
            # pdb.set_trace()
            for timestamp, group in procedure_data_grouped:
                proc_file.write(str(timestamp))                
                proc_file.write(',')
                
                current_med_ids = group.loc[group['proc_id'].notna()]['proc_id'].values.tolist()
                if current_med_ids:
                    proc_file.write(','.join([str(x) for x in current_med_ids]))
                    proc_file.write(',')
                proc_file.write('EOV,')
            # pdb.set_trace()
            proc_file.write('\n')
            patient_num += 1
            # if patient_num%logging_milestone==0:
            #     logging.info('Completed extracting and writing the diagnoses stream for {} patient with ENROLID = {}.'.format(cohort, current_id_diag))
        print('Finished processing all {} of procedure records for the cohort in {}'.format(total_num_records, (time.time() - proc_start_time)))            
        # logging.info('Finished processing all {} of diagnoses records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
        # logging.info('================================')    
    # return 0      
