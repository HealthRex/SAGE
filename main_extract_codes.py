import pdb
import argparse
import sys
import os
import utils.extract_codes as ext_cd
import logging

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--table_type", type=str, default='diagnosis', choices = ['diagnosis', 'medication', 'procedure', 'demographic'])    
parser.add_argument("--cohort", type=str, default='mci', choices = ['mci', 'nonmci'])    

parser.add_argument("--med_patient_id_field_name", type=str, default='anon_id')    
parser.add_argument("--med_code_field_name", type=str, default='pharm_class_abbr')    
parser.add_argument("--med_time_field_name", type=str, default='order_time_jittered') 

parser.add_argument("--proc_patient_id_field_name", type=str, default='anon_id')    
parser.add_argument("--proc_time_field_name", type=str, default='ordering_date_jittered')    
parser.add_argument("--proc_code_field_name", type=str, default='proc_id')    

parser.add_argument("--icd10_field_name", type=str, default='icd10')    
parser.add_argument("--icd9_field_name", type=str, default='icd9')    
parser.add_argument("--diag_time_field_name", type=str, default='start_date_utc')    
parser.add_argument("--diag_patient_id_field_name", type=str, default='anon_id')    




parser.add_argument("--display_step", type=int, default=100000)    

client_name ="mining-clinical-decisions"
# parser.add_argument("--logging_milestone", type=int, default=1000)    
# logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.INFO, filename = 'log/logfile_extract_streams.log', filemode = 'a')

if  parser.parse_args().table_type == "diagnosis" and parser.parse_args().cohort == 'mci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf2.mci_cohort_metadata_diagnosis` A ORDER BY A.anon_id, A.start_date_utc"
    icd_to_ccs_table_query = "SELECT * FROM `mining-clinical-decisions.mapdata.ahrq_ccsr_diagnosis`"
    icd9_to_icd10_query = "SELECT * FROM `mining-clinical-decisions.mapdata.icd9gem`"
    ext_cd.extract_diagnosis(client_name
                            , query_string
                            , icd_to_ccs_table_query
                            , icd9_to_icd10_query
                            , parser.parse_args().cohort
                            , parser.parse_args().icd10_field_name
                            , parser.parse_args().icd9_field_name
                            , parser.parse_args().diag_time_field_name
                            , parser.parse_args().diag_patient_id_field_name                            
                            , parser.parse_args().display_step)
elif  parser.parse_args().table_type == "medication" and parser.parse_args().cohort == 'mci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf2.mci_cohort_metadata_order_med` A ORDER BY A.anon_id, A.order_time_jittered"
    unique_medication_id_query = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf.unique_pharm_class_abbr` A WHERE A.pharm_class_abbr is not NULL"
    ext_cd.extract_medication(client_name
                            , query_string
                            , unique_medication_id_query
                            , parser.parse_args().cohort
                            , parser.parse_args().med_code_field_name
                            , parser.parse_args().med_time_field_name
                            , parser.parse_args().med_patient_id_field_name                                                        
                            , parser.parse_args().display_step)

elif  parser.parse_args().table_type == "procedure" and parser.parse_args().cohort == 'mci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf2.mci_cohort_metadata_order_proc` A ORDER BY A.anon_id, A.ordering_date_jittered"
    unique_proc_id_query = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf.unique_proc_ids` A WHERE A.proc_id is not NULL"    
    num_lines_query = "SELECT COUNT(*) FROM `mining-clinical-decisions.proj_sage_sf2.mci_cohort_metadata_order_proc`"
    ext_cd.extract_procedure(client_name
                            , query_string
                            , unique_proc_id_query
                            , num_lines_query
                            , parser.parse_args().cohort
                            , parser.parse_args().proc_code_field_name
                            , parser.parse_args().proc_time_field_name
                            , parser.parse_args().proc_patient_id_field_name                             
                            , parser.parse_args().display_step)


elif  parser.parse_args().table_type == "diagnosis" and parser.parse_args().cohort == 'nonmci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_metadata_age_matched_diagnosis` A ORDER BY A.anon_id, A.start_date_utc"
    icd_to_ccs_table_query = "SELECT * FROM `mining-clinical-decisions.ahrq_ccsr.diagnosis_code`"
    icd9_to_icd10_query = "SELECT * FROM `mining-clinical-decisions.mapdata.icd9gem`"
    ext_cd.extract_diagnosis(client_name
                            , query_string
                            , icd_to_ccs_table_query
                            , icd9_to_icd10_query
                            , parser.parse_args().cohort
                            , parser.parse_args().icd10_field_name
                            , parser.parse_args().icd9_field_name
                            , parser.parse_args().diag_time_field_name
                            , parser.parse_args().diag_patient_id_field_name                            
                            , parser.parse_args().display_step)


elif  parser.parse_args().table_type == "medication" and parser.parse_args().cohort == 'nonmci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_metadata_age_matched_order_med` A ORDER BY A.anon_id, A.order_time_jittered"
    unique_medication_id_query = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf.unique_pharm_class_abbr` A WHERE A.pharm_class_abbr is not NULL"    
    ext_cd.extract_medication(client_name
                            , query_string
                            , unique_medication_id_query
                            , parser.parse_args().cohort
                            , parser.parse_args().med_code_field_name
                            , parser.parse_args().med_time_field_name
                            , parser.parse_args().med_patient_id_field_name                                                        
                            , parser.parse_args().display_step)

elif  parser.parse_args().table_type == "procedure" and parser.parse_args().cohort == 'nonmci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_metadata_age_matched_order_proc` A ORDER BY A.anon_id, A.ordering_date_jittered"
    unique_proc_id_query = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf.unique_proc_ids` A WHERE A.proc_id is not NULL"        
    num_lines_query = "SELECT COUNT(*) FROM `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_metadata_age_matched_order_proc`"
    ext_cd.extract_procedure(client_name
                            , query_string
                            , unique_proc_id_query
                            , num_lines_query
                            , parser.parse_args().cohort
                            , parser.parse_args().proc_code_field_name
                            , parser.parse_args().proc_time_field_name
                            , parser.parse_args().proc_patient_id_field_name                             
                            , parser.parse_args().display_step)