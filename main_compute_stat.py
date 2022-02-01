import pdb
import argparse
import sys
import os
import utils.compute_stats_from_bg as comp_pt_nums
import logging

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--cohort_name", type=str, default='mci_all', choices = ['mci_all','non_mci_all_visited_neurology'])    
args = parser.parse_args()

# pdb.set_trace()
if  args.cohort_name == "mci_all":
	client_name = "mining-clinical-decisions"
	patient_id = 'anon_id'
	time_field_name = 'start_date_utc'
	query_diag = "select * from `mining-clinical-decisions.proj_sage_sf.mci_all_diagnosis` A ORDER BY A.anon_id"
	query_demog = "select * from `mining-clinical-decisions.proj_sage_sf.mci_all_demographic` "
	comp_pt_nums.compute_paients_numbers(args.cohort_name, client_name, patient_id, time_field_name, query_diag, query_demog)
# elif  args.cohort_name == "mci_referral":
# 	client_name = "mining-clinical-decisions"
# 	patient_id = 'rit_uid'
# 	query_diag = "select * from `mining-clinical-decisions.proj_sage_sf.mci_referral_diagnosis` A ORDER BY A.anon_id LIMIT 10000"
# 	query_demog = "select * from `mining-clinical-decisions.proj_sage_sf.mci_referral_demographic` "
# 	comp_pt_nums.compute_paients_numbers(args.cohort_name, client_name, patient_id, query_diag, query_demog)

elif  args.cohort_name == "non_mci_all_visited_neurology":
	client_name = "mining-clinical-decisions"
	patient_id = 'anon_id'
	time_field_name = 'start_date_utc'
	query_diag = "select * from `mining-clinical-decisions.proj_sage_sf.non_mci_all_visited_neurology_diagnosis` A ORDER BY A.anon_id"
	query_demog = "select * from `mining-clinical-decisions.proj_sage_sf.non_mci_all_visited_neurology_demographic` "
	comp_pt_nums.compute_paients_numbers(args.cohort_name, client_name, patient_id, time_field_name, query_diag,query_demog)
