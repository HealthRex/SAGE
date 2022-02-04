import pdb

def apply_exclusion_criteria(cohort
							,stationary_data_path
							,prediction_window_size
							,prior_clean_window_size	
							):
	pdb.set_trace()
	with open(stationary_data_path) as data_file, open('stationary_data/stationary_data_eligible_'+cohort+'.csv', 'w') as elig_file, open('stationary_data/stationary_data_ueligible_'+cohort+'.csv', 'w') as unelig_file:
		for line in data_file:

			print('Test')
