from google.cloud import bigquery;
from google.cloud.bigquery import dbapi;
import pandas as pd
import pdb
from dateutil.relativedelta import relativedelta

def compute_paients_numbers(cohort_name, client_name, patient_id , time_field_name, query_diag, query_demog, followup_window_size):
    # pdb.set_trace()
    client = bigquery.Client(client_name); 
    conn = dbapi.connect(client);
    cursor = conn.cursor();

    print('Executing SQL query to extract diagnosis_data records ...')
    # cursor.execute(query_diag);
    diagnosis_data = pd.read_sql_query(query_diag, conn);
    diagnosis_data_grouped = diagnosis_data.groupby(by=patient_id)

    print('Executing SQL query to extract demographic records ...')
    # cursor.execute(quenry_demog);
    demographic_data = pd.read_sql_query(query_demog, conn);

    # pdb.set_trace()

    metadata_columns=[patient_id, 'num_records', 'first_record_date', 'index_date_OR_diag_date', 'last_record_date', 'sex', 'bdate', 'canonical_race', 'MCI_label']
    metadata_list = []
    print('Extracting demographics and labeling the cohort ...')
    for id,group in diagnosis_data_grouped:
        # if id=='JC2a00006':
        #     pdb.set_trace()
        current_demog = demographic_data[demographic_data[patient_id]==id]
        group = group.sort_values(by=time_field_name,ascending=True)    
        num_records = len(group)
        first_record_date = group[time_field_name].iloc[0]
        last_record_date = group[time_field_name].iloc[-1]
        
        diag_date = last_record_date + relativedelta(months= -followup_window_size)

        sex = current_demog['GENDER'].iloc[0]
        bdate = current_demog['BIRTH_DATE_JITTERED'].iloc[0]
        canonical_race = current_demog['CANONICAL_RACE'].iloc[0]
        mci_records = group[(group['icd10'] == 'G31.84') | (group['icd10'] == 'F09') | (group['icd9'] == '331.83') | (group['icd9'] == '294.9')]
        MCI_label = 0
        if len(mci_records) > 0:
            # The patient has at least one MCI diagnoses
            current_patient_diag_date = mci_records[time_field_name].iloc[0]
            # pdb.set_trace()
            diag_date = current_patient_diag_date
            MCI_label = 1
            # print('MCI-positive')
        metadata_list.append([id, num_records, first_record_date, diag_date, last_record_date, sex, bdate, canonical_race, MCI_label])  

    metadata_pd = pd.DataFrame(metadata_list, columns=metadata_columns)
    metadata_pd.to_csv('intermediate_files/'+cohort_name+'_metadata.csv', index=False)
    
    