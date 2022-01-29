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
                    , display_step):
    pdb.set_trace()
    # end_of_diag_file = False
    # execute diagnosis query
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
    
    
    # while line_diag.split(',')[enrolid_idx] ==  'NULL' or line_diag.split(',')[enrolid_idx] ==  '':
    #     line_diag = diags_raw_file.readline().replace('\n','').replace("'","").rstrip('\n\\n\r\\r')
    # line_diag = line_diag.split(',')

    with open('intermediate_files/diagnosis_codes.csv', 'w') as diag_file:
        diag_file.write('patient id, timestamp, icd10 and icd9 codes, end of visit token')
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

            pdb.set_trace()
            current_patient_diags_pd = pd.DataFrame(np.array(current_patient_diags), columns= field_names)
            diagnosis_data_grouped = current_patient_diags_pd.groupby(by='timestamp_utc')
            # current_patient_diags_ar = np.array(current_patient_diags, dtype='U')       
            diag_file.write(current_id_diag)
            diag_file.write('\n')

            for timestamp, group in diagnosis_data_grouped:
                diag_file.write(timestamp)
                


                current_icd10s = ['ICD10_'+x for x in group.loc[group['icd10'].notna()]['icd10'].values.tolist()]
                diag_file.write(','.join(current_icd10s))
                diag_file.write(',')

                current_icd9s = ['ICD9_'+x for x in group.loc[group['icd9'].notna()]['icd9'].values.tolist()]
                diag_file.write(','.join(current_icd9s))
                diag_file.write(',')

            diag_file.write()
            
            diagnoses_file.write('\n')
            patient_num += 1
            if patient_num%logging_milestone==0:
                logging.info('Completed extracting and writing the diagnoses stream for {} patient with ENROLID = {}.'.format(cohort, current_id_diag))
        print('Finished processing all {} of diagnoses records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
        logging.info('Finished processing all {} of diagnoses records for the {} cohort in {}'.format(total_num_records, cohort, (time.time() - start_time)))            
        logging.info('================================')    
    # return 0      
