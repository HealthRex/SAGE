import pdb
import argparse
import sys
import os
import utils.create_metadata as comp_pt_nums
import logging

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--cohort_name", type=str, default='mci', choices = ['mci','nonmci'])    
parser.add_argument("--followup_window_size", type=int, default=6)    
args = parser.parse_args()

# pdb.set_trace()
if  args.cohort_name == "mci":
	client_name = "mining-clinical-decisions"
	patient_id = 'anon_id'
	time_field_name = 'start_date_utc'
	# query_diag = "select * from `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_mci_diagnosis` A ORDER BY A.anon_id"
	# query_demog = "select * from `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_mci_demographic` "
	query_diag = "select * from `mining-clinical-decisions.proj_sage_sf.mci_all_cohort_diagnosis` A "
	query_demog = "select * from `mining-clinical-decisions.proj_sage_sf.mci_all_cohort_demographic` "
	comp_pt_nums.compute_paients_numbers(args.cohort_name, client_name, patient_id, time_field_name, query_diag, query_demog,args.followup_window_size)

elif  args.cohort_name == "nonmci":
	client_name = "mining-clinical-decisions"
	patient_id = 'anon_id'
	time_field_name = 'start_date_utc'
	# query_diag = "select * from `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci_diagnosis` A ORDER BY A.anon_id"
	# query_demog = "select * from `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci_demographic` "
	query_diag = "select * from `mining-clinical-decisions.proj_sage_sf.nonmci_all_cohort_sampled2_diagnosis` A "
	query_demog = "select * from `mining-clinical-decisions.proj_sage_sf.nonmci_all_cohort_sampled2_demographic` "	
	comp_pt_nums.compute_paients_numbers(args.cohort_name, client_name, patient_id, time_field_name, query_diag,query_demog, args.followup_window_size)
