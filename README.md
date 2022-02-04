<h1 style="font-size:60px;">1. Extracting case and control cohorts using big query scripts</h1>

Use the following script to extract all patients who have visited neurology department as a new patient and save the results ina big query table ```all_new_patients_in_neurology``` (at the moment 8,496 patients). 

```
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
```
From the cohort generated using the above script, you can select patients who have never had any MCI diagnosis code using:
```
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
```
This (at the moment) resulted in 7,875 patients and we saved them under ```all_new_patients_in_neurology_nonmci``` table. Then, use the following scripts to extract mci patients within the cohort of new patients in neurology:
```
(SELECT DISTINCT A.anon_id
    FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology` A
    )
EXCEPT DISTINCT 
(
SELECT DISTINCT B.anon_id
    FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` B
)
```
This resulted in 621 patients. The results were saved under ```all_new_patients_in_neurology_mci```. 

Extract diagnosis, medication, procedure and demographic records for non-mci patients in ```all_new_patients_in_neurology_nonmci```:
```
-- Diagnosis saved under all_new_patients_in_neurology_nonmci_diagnosis
SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` N
INNER JOIN `mining-clinical-decisions.shc_core.diagnosis_code` DG
ON N.anon_id = DG.anon_id

-- Saved under all_new_patients_in_neurology_nonmci_order_med
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` N
INNER JOIN `mining-clinical-decisions.shc_core.order_med` M
ON N.anon_id = M.anon_id

-- Saved under all_new_patients_in_neurology_nonmci_order_proc
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` N
INNER JOIN `mining-clinical-decisions.shc_core.order_proc` P
ON N.anon_id = P.anon_id

-- Demographics saved under all_new_patients_in_neurology_nonmci_demographic
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology_nonmci` N
LEFT JOIN `mining-clinical-decisions.shc_core.demographic` DM
ON N.anon_id = DM.anon_id
```
