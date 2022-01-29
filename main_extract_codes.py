import pdb
import argparse
import sys
import os
import utils.extract_codes as ext_cd
import logging

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--table_type", type=str, default='diagnosis', choices = ['diagnosis', 'medication', 'procedure', 'demographic'])    
parser.add_argument("--cohort", type=str, default='mci', choices = ['mci', 'non_mci'])    

parser.add_argument("--display_step", type=int, default=100000)    

client_name ="mining-clinical-decisions"
# parser.add_argument("--logging_milestone", type=int, default=1000)    
# logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.INFO, filename = 'log/logfile_extract_streams.log', filemode = 'a')

if  parser.parse_args().table_type == "diagnosis" and parser.parse_args().cohort == 'mci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf.mci_all_diagnosis` A ORDER BY A.jc_uid, A.timestamp_utc LIMIT 10000"
    ext_cd.extract_diagnosis(client_name
                            , query_string
                            , parser.parse_args().cohort
                            , parser.parse_args().display_step)

elif  parser.parse_args().table_type == "diagnosis" and parser.parse_args().cohort == 'non_mci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf.non_mci_pc_referral_diagnosis` A ORDER BY A.jc_uid, A.timestamp_utc LIMIT 10000"
    ext_cd.extract_diagnosis(client_name
                            , query_string
                            , parser.parse_args().cohort
                            , parser.parse_args().display_step)


elif  parser.parse_args().table_type == "medication" and parser.parse_args().cohort == 'non_mci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf.non_mci_pc_referral_order_med` A ORDER BY A.jc_uid, A.order_time_jittered_utc LIMIT 10000"
    ext_cd.extract_medication(client_name
                            , query_string
                            , parser.parse_args().cohort
                            , parser.parse_args().display_step)

elif  parser.parse_args().table_type == "procedure" and parser.parse_args().cohort == 'non_mci':
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf.non_mci_pc_referral_order_proc` A ORDER BY A.jc_uid, A.order_time_jittered_utc LIMIT 100000"
    ext_cd.extract_procedure(client_name
                            , query_string
                            , parser.parse_args().cohort
                            , parser.parse_args().display_step)
