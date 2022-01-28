-- This script creates MCI cohort: extract all patients with at least one MCI diagnosis in Diagnosis_code
SELECT D.jc_uid, MIN(D.timestamp_utc) AS diagnoses_date
FROM `mining-clinical-decisions.starr_datalake2018.diagnosis_code` D
WHERE (D.icd10 = 'G31.84'
   OR D.icd10 = 'F09'
   OR D.icd9 = '331.83'
   OR D.icd9 = '294.9')
GROUP BY D.jc_uid


-- Join the MCI cohort with diagnosis table
SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.mci_all_cohort` MC
INNER JOIN `mining-clinical-decisions.starr_datalake2018.diagnosis_code` DG
ON MC.jc_uid = DG.jc_uid

-- Join the MCI cohort with demographic table
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.mci_all_cohort` MC
INNER JOIN `mining-clinical-decisions.starr_datalake2018.demographic` DM
ON MC.jc_uid = DM.rit_uid

-- Join the MCI cohort with diagnosis table
SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.mci_all_cohort` MC
INNER JOIN `mining-clinical-decisions.starr_datalake2018.diagnosis_code` DG
ON MC.jc_uid = DG.jc_uid