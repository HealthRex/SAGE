-- Extracting everyone who has been visited in the Neurology department as a new patients: 8496 patients
-- Saved under all_visited_neurology_cohort
WITH
    SP AS
        (
            select enc.anon_id, enc.pat_enc_csn_id_coded as SP_enc, enc.appt_when_jittered as SP_app_datetime --, DX.icd10
            from `mining-clinical-decisions.shc_core.encounter` as enc 
        join `mining-clinical-decisions.shc_core.dep_map` as dep on enc.department_id = dep.department_id    
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

SELECT DISTINCT COHORT.anon_id
FROM COHORT 


-- From the above cohort (8496 patients), extracting anyone who never had any MCI diagnosis code: 7248 patients
-- saved in non_mci_all_visited_neurology_cohort
(SELECT DISTINCT D.anon_id
FROM `mining-clinical-decisions.shc_core.diagnosis_code` D
INNER JOIN `mining-clinical-decisions.proj_sage_sf.all_visited_neurology_cohort` R
ON D.anon_id = R.anon_id)
EXCEPT DISTINCT 
(
SELECT DiagT.anon_id
FROM `mining-clinical-decisions.shc_core.diagnosis_code` DiagT
WHERE (DiagT.icd10 = 'G31.84'
   OR DiagT.icd10 = 'F09'
   OR DiagT.icd9 = '331.83'
   OR DiagT.icd9 = '294.9')
GROUP BY DiagT.anon_id )



-- Diagnosis for controls
-- saved in non_mci_all_visited_neurology_diagnosis
SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.non_mci_all_visited_neurology_cohort` N
INNER JOIN `mining-clinical-decisions.shc_core.diagnosis_code` DG
ON N.anon_id = DG.anon_id


-- Demographic
-- saved in non_mci_all_visited_neurology_demographic
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.non_mci_all_visited_neurology_cohort` N
INNER JOIN `mining-clinical-decisions.shc_core.demographic` DM
ON N.anon_id = DM.anon_id

-- lab_result
-- saved in non_mci_all_visited_neurology_lab_result
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.non_mci_all_visited_neurology_cohort` N
INNER JOIN `mining-clinical-decisions.shc_core.lab_result` DM
ON N.anon_id = DM.anon_id

-- order_med
-- saved in non_mci_all_visited_neurology_order_med
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.non_mci_all_visited_neurology_cohort` N
INNER JOIN `mining-clinical-decisions.shc_core.order_med` M
ON N.anon_id = M.anon_id

-- order_proc
-- saved in on_mci_all_visited_neurology_order_proc
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.non_mci_all_visited_neurology_cohort` N
INNER JOIN `mining-clinical-decisions.shc_core.order_proc` P
ON N.anon_id = P.anon_id


--==================
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
