-- Extracting everyone who has been visited in the Neurology department as a new patients: 6219 patients
WITH
    SP AS
        (
            select enc.jc_uid, enc.pat_enc_csn_id_coded as SP_enc, enc.appt_when_jittered as SP_app_datetime --, DX.icd10
            from `starr_datalake2018.encounter` as enc 
        join `starr_datalake2018.dep_map` as dep on enc.department_id = dep.department_id    
        --join `starr_datalake2018.diagnosis_code` as DX on (enc.pat_enc_csn_id_coded = DX.pat_enc_csn_id_coded)
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
  SELECT SP.*
  FROM SP 
)

SELECT count(DISTINCT COHORT.jc_uid)
FROM COHORT 


-- Extracting everyone who has been referred to NEUROLOGY: 47164 patients
-- saved into all_referral_cohort
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

 COHORT AS
  (
  SELECT PC.*
  FROM PC 
)

SELECT DISTINCT COHORT.rit_uid
FROM COHORT 


-- From the above cohort (47164 patients), extracting anyone who never had any MCI diagnosis code: 47133 patients
-- saved in non_mci_pc_referral_cohort
(SELECT DISTINCT D.jc_uid
FROM `mining-clinical-decisions.starr_datalake2018.diagnosis_code` D
INNER JOIN `mining-clinical-decisions.proj_sage_sf.all_referral_cohort` R
ON D.jc_uid = R.rit_uid)
EXCEPT DISTINCT 
(
SELECT DiagT.jc_uid
FROM `mining-clinical-decisions.starr_datalake2018.diagnosis_code` DiagT
WHERE (DiagT.icd10 = 'G31.84'
   OR DiagT.icd10 = 'F09'
   OR DiagT.icd9 = '331.83'
   OR DiagT.icd9 = '294.9')
GROUP BY DiagT.jc_uid )



-- Diagnosis for controls
-- saved in non_mci_pc_referral_diagnosis
SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.non_mci_pc_referral_cohort` N
INNER JOIN `mining-clinical-decisions.starr_datalake2018.diagnosis_code` DG
ON N.jc_uid = DG.jc_uid


-- Demographic
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.non_mci_pc_referral_cohort` N
INNER JOIN `mining-clinical-decisions.starr_datalake2018.demographic` DM
ON N.jc_uid = DM.rit_uid