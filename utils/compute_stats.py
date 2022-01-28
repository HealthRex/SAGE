import os
import pdb
import pandas as pd
pdb.set_trace()

data = pd.read_csv('./data/sample_data.csv')
data_grouped = data.groupby(by='rit_uid')


metadata_columns=['rit_uid', 'num_records', 'first_record_date', 'diag_date', 'last_record_date', 'sex', 'bdate', 'canonical_race', 'MCI_label']
metadata_list = []
for id,group in data_grouped:
	group['timestamp_utc'] = pd.to_datetime(group['timestamp_utc'])
	group = group.sort_values(by='timestamp_utc',ascending=True)	
	num_records = len(group)
	first_record_date = group['timestamp_utc'].iloc[0]
	last_record_date = group['timestamp_utc'].iloc[-1]
	diag_date = last_record_date
	sex = group['gender'].iloc[0]
	bdate = group['birth_date_jittered'].iloc[0]
	canonical_race = group['canonical_race'].iloc[0]
	mci_records = group[(group['icd10'] == 'G31.84') | (group['icd10'] == 'F09') | (group['icd9'] == '331.83') | (group['icd9'] == '294.9')]
	MCI_label = 0
	if len(mci_records) > 0:
		# pdb.set_trace()
		# The patient has at least one MCI diagnoses
		current_patient_diag_date = mci_records['timestamp_utc'].iloc[0]
		# pdb.set_trace()
		diag_date = current_patient_diag_date
		MCI_label = 1
		# print('MCI-positive')
	metadata_list.append([id, num_records, first_record_date, diag_date, last_record_date, sex, bdate, canonical_race, MCI_label])	

metadata_pd = pd.DataFrame(metadata_list, columns=metadata_columns)
metadata_pd.to_csv('intermediate_files/cohort_metadata.csv', index=False)

pdb.set_trace()
print('The end')