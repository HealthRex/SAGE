I used this code to extract random controls:
SELECT * 
FROM `mining-clinical-decisions.proj_sage_sf.nonmci_all_cohort`  
ORDER BY RAND() 
LIMIT 150000

SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.nonmci_all_cohort_sampled2` N
INNER JOIN `mining-clinical-decisions.shc_core.diagnosis_code` DG
ON N.anon_id = DG.anon_id

SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.nonmci_all_cohort_sampled2` N
INNER JOIN `mining-clinical-decisions.shc_core.order_med` M
ON N.anon_id = M.anon_id

SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.nonmci_all_cohort_sampled2` N
INNER JOIN `mining-clinical-decisions.shc_core.order_proc` P
ON N.anon_id = P.anon_id

SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.nonmci_all_cohort_sampled2` N
LEFT JOIN `mining-clinical-decisions.shc_core.demographic` DM
ON N.anon_id = DM.anon_id

/*
WITH
    SP AS
        (
            select enc.anon_id, enc.pat_enc_csn_id_coded as SP_enc, enc.appt_when_jittered as SP_app_datetime --, DX.icd10
            from `mining-clinical-decisions.shc_core.encounter` as enc 
            join `mining-clinical-decisions.shc_core.dep_map` as dep on enc.department_id = dep.department_id    
            where dep.specialty_dep_c = '19' -- dep.specialty like '%NEUROLOGY%'
            AND visit_type like 'NEW PATIENT%' -- Naturally screens to only 'Office Visit' enc_type 
            AND appt_status = 'Completed'
        ),

 COHORT AS
  (
  SELECT SP.*
  FROM SP 
)

SELECT DISTINCT COHORT.anon_id
FROM COHORT 
*/
/*
(SELECT DISTINCT A.anon_id
    FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology` A
    )
EXCEPT DISTINCT 
(
SELECT DISTINCT B.anon_id
    FROM `mining-clinical-decisions.shc_core.diagnosis_code` B
    WHERE (B.icd10 = 'G31.84'
        OR B.icd10 = 'F09'
        OR B.icd9 = '331.83'
        OR B.icd9 = '294.9')
GROUP BY B.anon_id )
*/
/*
(SELECT DISTINCT A.anon_id
    FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology` A
    )
EXCEPT DISTINCT 
(
SELECT DISTINCT B.anon_id
    FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` B
)
*/
/*
SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` N
INNER JOIN `mining-clinical-decisions.shc_core.diagnosis_code` DG
ON N.anon_id = DG.anon_id
*/
/*
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` N
INNER JOIN `mining-clinical-decisions.shc_core.order_med` M
ON N.anon_id = M.anon_id
*/

/*
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` N
INNER JOIN `mining-clinical-decisions.shc_core.order_proc` P
ON N.anon_id = P.anon_id
*/

/*
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` N
LEFT JOIN `mining-clinical-decisions.shc_core.demographic` DM
ON N.anon_id = DM.anon_id
*/

/*
-- Diagnosis saved under all_new_patients_in_neurology_mci_diagnosis
SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_mci` N
INNER JOIN `mining-clinical-decisions.shc_core.diagnosis_code` DG
ON N.anon_id = DG.anon_id

-- Saved under all_new_patients_in_neurology_mci_order_med
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_mci` N
INNER JOIN `mining-clinical-decisions.shc_core.order_med` M
ON N.anon_id = M.anon_id


-- Saved under all_new_patients_in_neurology_mci_order_proc
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_mci` N
INNER JOIN `mining-clinical-decisions.shc_core.order_proc` P
ON N.anon_id = P.anon_id


-- Demographics saved under all_new_patients_in_neurology_mci_demographic
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_mci` N
LEFT JOIN `mining-clinical-decisions.shc_core.demographic` DM
ON N.anon_id = DM.anon_id

*/