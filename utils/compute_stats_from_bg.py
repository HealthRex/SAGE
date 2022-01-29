from google.cloud import bigquery;
from google.cloud.bigquery import dbapi;
import pandas as pd
import pdb


def compute_paients_numbers(cohort_name, client_name, patient_id ,query_diag, query_demog):
    # pdb.set_trace()
    client = bigquery.Client(client_name); 
    conn = dbapi.connect(client);
    cursor = conn.cursor();

    print('Executing SQL query to extract diagnosis_data records ...')
    cursor.execute(query_diag);
    diagnosis_data = pd.read_sql_query(query_diag, conn);
    diagnosis_data_grouped = diagnosis_data.groupby(by=patient_id)

    print('Executing SQL query to extract demographic records ...')
    cursor.execute(query_demog);
    demographic_data = pd.read_sql_query(query_demog, conn);

    # pdb.set_trace()

    metadata_columns=[patient_id, 'num_records', 'first_record_date', 'diag_date', 'last_record_date', 'sex', 'bdate', 'canonical_race', 'MCI_label']
    metadata_list = []
    print('Extracting demographics and labeling the cohort ...')
    for id,group in diagnosis_data_grouped:
        current_demog = demographic_data[demographic_data[patient_id]==id]
        group = group.sort_values(by='timestamp_utc',ascending=True)    
        num_records = len(group)
        first_record_date = group['timestamp_utc'].iloc[0]
        last_record_date = group['timestamp_utc'].iloc[-1]
        diag_date = last_record_date
        sex = current_demog['gender'].iloc[0]
        bdate = current_demog['birth_date_jittered'].iloc[0]
        canonical_race = current_demog['canonical_race'].iloc[0]
        mci_records = group[(group['icd10'] == 'G31.84') | (group['icd10'] == 'F09') | (group['icd9'] == '331.83') | (group['icd9'] == '294.9')]
        MCI_label = 0
        if len(mci_records) > 0:
            # The patient has at least one MCI diagnoses
            current_patient_diag_date = mci_records['timestamp_utc'].iloc[0]
            # pdb.set_trace()
            diag_date = current_patient_diag_date
            MCI_label = 1
            # print('MCI-positive')
        metadata_list.append([id, num_records, first_record_date, diag_date, last_record_date, sex, bdate, canonical_race, MCI_label])  

    metadata_pd = pd.DataFrame(metadata_list, columns=metadata_columns)
    metadata_pd.to_csv('intermediate_files/'+cohort_name+'_metadata.csv', index=False)
    