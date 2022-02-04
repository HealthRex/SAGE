<h1 style="font-size:60px;">1. Extracting case and control cohorts using big query scripts</h1>

Use the following script to extract patients with at least 1 diagnosis of MCI in their records:

```
SELECT D.anon_id, MIN(D.start_date_utc) as diagnosis_date
FROM `mining-clinical-decisions.shc_core.diagnosis_code` D
WHERE (D.icd10 = 'G31.84'
   OR D.icd10 = 'F09'
   OR D.icd9 = '331.83'
   OR D.icd9 = '294.9')
GROUP BY D.anon_id

```
