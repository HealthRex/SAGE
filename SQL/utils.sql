-- Finding unique ICD10 codes
-- Saved in unique_icd10s

SELECT DISTINCT A.icd10
FROM `mining-clinical-decisions.shc_core.diagnosis_code` A


SELECT DISTINCT A.icd9
FROM `mining-clinical-decisions.shc_core.diagnosis_code` A


SELECT DISTINCT A.medication_id
FROM `mining-clinical-decisions.shc_core.order_med` A


SELECT DISTINCT A.proc_id
FROM `mining-clinical-decisions.shc_core.order_proc` A