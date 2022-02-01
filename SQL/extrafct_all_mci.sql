-- This script creates MCI cohort: extract all patients with at least one MCI diagnosis in Diagnosis_code
-- Resulted in 11,528 patients and saved them in mci_all_cohort
SELECT D.anon_id, MIN(D.start_date_utc) as diagnosis_date
FROM `mining-clinical-decisions.shc_core.diagnosis_code` D
WHERE (D.icd10 = 'G31.84'
   OR D.icd10 = 'F09'
   OR D.icd9 = '331.83'
   OR D.icd9 = '294.9')
GROUP BY D.anon_id


-- Join the MCI cohort with diagnosis table
-- Resulted in 4773155 records and saved in mci_all_diagnosis
SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.mci_all_cohort` MC
INNER JOIN `mining-clinical-decisions.shc_core.diagnosis_code` DG
ON MC.anon_id = DG.anon_id

-- Join the MCI cohort with demographic table
-- Resulted in 11528 records and saved them in 
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.mci_all_cohort` MC
INNER JOIN `mining-clinical-decisions.shc_core.demographic` DM
ON MC.anon_id = DM.anon_id


-- lab_result
-- Resulted in 12174861 and saved in mci_all_lab_result
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.mci_all_cohort` C
INNER JOIN `mining-clinical-decisions.shc_core.lab_result` L
ON C.anon_id = L.anon_id

-- order_med
-- Resulted in  2146981 and saved in mci_all_order_med
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.mci_all_cohort` C
INNER JOIN `mining-clinical-decisions.shc_core.order_med` M
ON C.anon_id = M.anon_id

-- order_proc
-- saved in mci_all_order_proc
SELECT *
FROM `mining-clinical-decisions.proj_sage_sf.mci_all_cohort` C
INNER JOIN `mining-clinical-decisions.shc_core.order_proc` P
ON C.anon_id = P.anon_id