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
(SELECT DISTINCT D.anon_id
    FROM `mining-clinical-decisions.shc_core.diagnosis_code` D
    INNER JOIN `mining-clinical-decisions.proj_sage_sf.all_new_patients_in_neurology` R
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
```
This (at the moment) resulted in 7,248 patients and we saved them under ```all_new_patients_in_neurology_nonmci``` table. 
