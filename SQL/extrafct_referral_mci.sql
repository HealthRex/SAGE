-- Extracting referral cohort

WITH
  PC AS
  (
    select DM.* ,op.jc_uid, op.pat_enc_csn_id_coded as PC_enc, 
        enc.appt_when_jittered as PC_app_datetime, op.order_time_jittered as PC_ref_datetime,
        (case when DM.gender='Male' then 0 else 1 end) as gender_bool, 
        DATETIME_DIFF( CAST(enc.appt_when_jittered as DATETIME) ,CAST( DM.birth_date_jittered as DATETIME), YEAR) as age
    from `mining-clinical-decisions.starr_datalake2018.order_proc` as op 
      join `mining-clinical-decisions.starr_datalake2018.encounter` as enc on op.pat_enc_csn_id_coded = enc.pat_enc_csn_id_coded 
      join `mining-clinical-decisions.starr_datalake2018.demographic` as DM on (op.jc_uid = DM.rit_uid)
      --join `mining-clinical-decisions.starr_datalake2018.diagnosis_code` as DX on (enc.pat_enc_csn_id_coded = DX.pat_enc_csn_id_coded)
    WHERE proc_code LIKE 'REF%'and description LIKE '%NEUROLOGY%'
    and ordering_mode = 'Outpatient'
  ),
  
  SP AS
  (
    select enc.jc_uid, enc.pat_enc_csn_id_coded as SP_enc, enc.appt_when_jittered as SP_app_datetime --, DX.icd10
    from `mining-clinical-decisions.starr_datalake2018.encounter` as enc 
    join `mining-clinical-decisions.starr_datalake2018.dep_map` as dep on enc.department_id = dep.department_id    
    --join `mining-clinical-decisions.starr_datalake2018.diagnosis_code` as DX on (enc.pat_enc_csn_id_coded = DX.pat_enc_csn_id_coded)
    where 
    dep.specialty_dep_c = '19' -- dep.specialty like '%NEUROLOGY%'
    AND visit_type like 'NEW PATIENT%' -- Naturally screens to only 'Office Visit' enc_type 
    -- and appt_type in ('Office Visit','Appointment') -- Otherwise Telephone, Refill, Orders Only, etc.
    AND appt_status = 'Completed'
    --AND (
    --icd10 LIKE 'G31.84' -- Mild cognitive impairment
    --OR icd9 LIKE '331.83'
    --OR icd10 LIKE 'G30%' -- ANY Type of Alzheimer's disease
    --OR icd9 LIKE '331.0'
    --)
  ),
  
  COHORT AS
  (
  SELECT PC.*, SP.* EXCEPT (jc_uid)
  FROM PC JOIN SP USING (jc_uid)
  WHERE SP.SP_app_datetime BETWEEN PC.PC_ref_datetime AND DATETIME_ADD(PC.PC_ref_datetime, INTERVAL 4 MONTH)
  ORDER BY PC.jc_uid 
)
SELECT * FROM COHORT

-- Extracting diagnosis 
SELECT * FROM COHORT
INNER JOIN `mining-clinical-decisions.starr_datalake2018.diagnosis_code` DG
ON COHORT.jc_uid = DG.jc_uid

-- Extracting demographic
SELECT * FROM COHORT
INNER JOIN `mining-clinical-decisions.starr_datalake2018.demographic` DM
ON COHORT.jc_uid = DM.rit_uid