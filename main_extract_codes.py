import pdb
import argparse
import sys
import os
import utils.extract_codes as ext_cd
import logging

sys.path.append(os.getcwd())
parser = argparse.ArgumentParser()
parser.add_argument("--table_type", type=str, default='diagnosis', choices = ['diagnosis', 'medication', 'procedure', 'demographic'])    
parser.add_argument("--display_step", type=int, default=100000)    

client_name ="mining-clinical-decisions"
# parser.add_argument("--logging_milestone", type=int, default=1000)    
# logging.basicConfig(format='Date-Time : %(asctime)s : Line No. : %(lineno)d - %(message)s', level = logging.INFO, filename = 'log/logfile_extract_streams.log', filemode = 'a')

if  parser.parse_args().table_type == "diagnosis":
    query_string = "SELECT * FROM `mining-clinical-decisions.proj_sage_sf.mci_all_diagnosis` A ORDER BY A.jc_uid, A.timestamp_utc LIMIT 1000"
    ext_cd.extract_diagnosis(client_name
                            , query_string
                            , parser.parse_args().display_step)

