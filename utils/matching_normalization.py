import pdb
import pandas as pd
from sklearn.model_selection import train_test_split

def matching(mci_stationary_data_path
			,non_mci_stationary_data_path			
			,mci_metadata_path
			,non_mci_metadata_path
			,case_control_ratio
			,matching
			):
	# pdb.set_trace()
	print('Matching cases and controls. Imbalance ratio is {}'.format(case_control_ratio))
	
	mci_data = pd.read_csv(mci_stationary_data_path)
	mci_data.columns = mci_data.columns.str.strip()
	non_mci_data = pd.read_csv(non_mci_stationary_data_path)
	non_mci_data.columns = non_mci_data.columns.str.strip()
	# Match based on age and sex
	case_metadata = pd.read_csv(mci_metadata_path)
	control_metadata = pd.read_csv(non_mci_metadata_path)
	control_metadata = control_metadata[control_metadata['anon_id'].isin(non_mci_data['Patient_ID'].values)]

	control_metadata['matched'] = 0

	control_metadata['byear'] = control_metadata['bdate'].str[:4]
	case_metadata['byear'] = case_metadata['bdate'].str[:4]

	# for idx, row in case_metadata.iterrows():
	for idx, row in mci_data.iterrows():
		current_patient_metadata = case_metadata[case_metadata['anon_id']==row['Patient_ID']]
		if len(current_patient_metadata) !=1:
			pdb.set_trace()

		if current_patient_metadata['sex'].values[0] != 'Unknown':
			matched_controls = control_metadata.loc[(control_metadata['sex'] == current_patient_metadata['sex'].values[0]) & (control_metadata['byear'] == current_patient_metadata['byear'].values[0]) & (control_metadata['matched'] ==0)]
		else:			
			matched_controls = control_metadata.loc[(control_metadata['byear'] == current_patient_metadata['byear'].values[0]) & (control_metadata['matched'] ==0)]

		if matched_controls.shape[0] >= case_control_ratio:
			control_metadata.loc[matched_controls.index[:case_control_ratio], 'matched'] = 1
		else:			
			sex_matched_controls = control_metadata.loc[(control_metadata['sex'] == current_patient_metadata['sex'].values[0]) & (control_metadata['matched'] ==0)]
			matched_controls = sex_matched_controls.iloc[(sex_matched_controls['byear'].astype(int) -int(current_patient_metadata['byear'].values[0])).abs().argsort()[:1]]	
			if matched_controls.shape[0] >= case_control_ratio:
				control_metadata.loc[matched_controls.index[:case_control_ratio], 'matched'] = 1
			else:
				pdb.set_trace()
				print('Couldnt find any match!')	
		# print('Test')
	# pdb.set_trace()	

	control_metadata_matched = control_metadata[control_metadata['matched']==1]
	control_metadata_matched.to_csv(non_mci_metadata_path[:-4]+'_matched.csv', index=False)
	# pdb.set_trace()
	non_mci_data_matched = non_mci_data[non_mci_data['Patient_ID'].isin(control_metadata_matched['anon_id'].values.tolist())]
	for i in range(non_mci_data_matched.shape[1]):
		if non_mci_data_matched.columns[i].strip() != mci_data.columns[i].strip():
			pdb.set_trace()
	# pdb.set_trace()	
	# non_mci_data_matched = non_mci_data_matched.reindex(columns=mci_data.columns)
	if non_mci_data_matched.shape[1] != mci_data.shape[1]:
		pdb.set_trace()
		print('Case and control dimensions do not match')
	
	# pdb.set_trace()
	
	# if non_mci_data_matched.shape[0] > mci_data.shape[0]:
	# 	non_mci_data_matched = non_mci_data_matched.sample(n=mci_data.shape[0])
	# else:
	# 	pdb.set_trace()
	# 	print('Warning: matched control population size is smaller than control population size.')	
	all_data = mci_data.append(non_mci_data_matched, ignore_index=True).sample(frac=1).reset_index(drop=True)

	all_data.to_csv('stationary_data/stationary_data_imbratio'+str(case_control_ratio)+'.csv', index=False)

	return 'stationary_data/stationary_data_imbratio'+str(case_control_ratio)+'.csv'

def normalization(data_path
				,test_ratio
				):
	# pdb.set_trace()
	epsil = 2.220446049250313e-16
	round_precision = 5
	data = pd.read_csv(data_path)

	mins = data.iloc[:,1:].min()
	maxes = data.iloc[:,1:].max()

	normalized_data=(data.iloc[:,1:] -mins)/((maxes-mins) + epsil)
	normalized_data['Patient_ID'] = data['Patient_ID']
	normalized_data['Label'] = data['Label']
	normalized_data.round(round_precision).to_csv(data_path[:-4]+'_normalized.csv', index=False)

	trainset, testset = train_test_split(normalized_data, test_size = test_ratio, shuffle=False)

	trainset.to_csv(data_path[:-4]+'_normalized_train.csv', index=False)
	testset.to_csv(data_path[:-4]+'_normalized_test.csv', index=False)
