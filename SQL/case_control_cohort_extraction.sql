-- Find everyone in the data with at least one MCI and save them in mining-clinical-decisions.proj_sage_sf2.mci_all_cohort
SELECT D.anon_id, MIN(D.start_date_utc) AS diagnoses_date
FROM `mining-clinical-decisions.shc_core.diagnosis_code` D
WHERE (D.icd10 = 'G31.84'
   OR D.icd10 = 'F09'
   OR D.icd9 = '331.83'
   OR D.icd9 = '294.9')
GROUP BY D.anon_id

-- Find mci patient with at least 2 years of data before diagnosis date and extract demog data for them 
-- and save it under proj_sage_sf2.mci_cohort_metadata
select X.anon_id
    , X.first_Record_date
    , Y.diagnoses_date
    , X.last_record_date
    , DEM.BIRTH_DATE_JITTERED
    , DEM.GENDER
    , DEM.CANONICAL_RACE
from 
(select A.anon_id
    , MIN(D.start_date_utc) as first_Record_date
    , MAX(D.start_date_utc) as last_record_date
from `mining-clinical-decisions.proj_sage_sf2.mci_cohort_all` A
inner join `mining-clinical-decisions.shc_core.diagnosis_code` D on A.anon_id = D.anon_id
group by A.anon_id) X
inner join `mining-clinical-decisions.proj_sage_sf2.mci_cohort_all` Y on X.anon_id = Y.anon_id
inner join `mining-clinical-decisions.shc_core.demographic` DEM on Y.anon_id = DEM.anon_id
where TIMESTAMP_DIFF(Y.diagnoses_date, X.first_Record_date, DAY) > 730


-- Find all nonmci patients and save them in nonmci_cohort_all
(select distinct A.anon_id
from `mining-clinical-decisions.shc_core.diagnosis_code` A)
EXCEPT DISTINCT
(select distinct B.anon_id
from `mining-clinical-decisions.proj_sage_sf2.mci_cohort_all` B)


-- Metadata for control cohort
select X.anon_id
    , X.first_record_date
    , X.last_record_date
    , DEM.BIRTH_DATE_JITTERED
    , DEM.GENDER
    , DEM.CANONICAL_RACE
from
(select A.anon_id
    , MIN(D.start_date_utc) as first_record_date
    , MAX(D.start_date_utc) as last_record_date
from `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_all` A
inner join `mining-clinical-decisions.shc_core.diagnosis_code` D on D.anon_id = A.anon_id
group by A.anon_id) X
inner join `mining-clinical-decisions.shc_core.demographic` DEM on DEM.anon_id = X.anon_id
where TIMESTAMP_DIFF(X.last_record_date, X.first_record_date, DAY) > 1095 -- 3 years


select *
from `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_metadata` A
where A.BIRTH_DATE_JITTERED in
(select B.BIRTH_DATE_JITTERED
from `mining-clinical-decisions.proj_sage_sf2.mci_cohort_metadata` B) 


SELECT DG.*
FROM `mining-clinical-decisions.proj_sage_sf.nonmci_cohort_matched` N
INNER JOIN `mining-clinical-decisions.shc_core.diagnosis_code` DG
ON N.anon_id = DG.anon_id



-- extract data for mci 
-- mci diagnosis
SELECT B.*
FROM `mining-clinical-decisions.proj_sage_sf2.mci_cohort_metadata` A
INNER JOIN `mining-clinical-decisions.shc_core.diagnosis_code` B
ON A.anon_id = B.anon_id

-- mci meds
SELECT B.*
FROM `mining-clinical-decisions.proj_sage_sf2.mci_cohort_metadata` A
INNER JOIN `mining-clinical-decisions.shc_core.order_med` B
ON A.anon_id = B.anon_id

-- mci procs
SELECT B.*
FROM `mining-clinical-decisions.proj_sage_sf2.mci_cohort_metadata` A
INNER JOIN `mining-clinical-decisions.shc_core.order_proc` B
ON A.anon_id = B.anon_id

-- mci demogs
SELECT B.*
FROM `mining-clinical-decisions.proj_sage_sf2.mci_cohort_metadata` A
INNER JOIN `mining-clinical-decisions.shc_core.demographic` B
ON A.anon_id = B.anon_id


-- extract data for controls
-- nonmci diagnosis
SELECT B.*
FROM `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_metadata_age_matched` A
INNER JOIN `mining-clinical-decisions.shc_core.diagnosis_code` B
ON A.anon_id = B.anon_id

-- nonmci meds
SELECT B.*
FROM `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_metadata_age_matched` A
INNER JOIN `mining-clinical-decisions.shc_core.order_med` B
ON A.anon_id = B.anon_id

-- nonmci procs
SELECT B.*
FROM `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_metadata_age_matched` A
INNER JOIN `mining-clinical-decisions.shc_core.order_proc` B
ON A.anon_id = B.anon_id

-- nonmci demogs
SELECT B.*
FROM `mining-clinical-decisions.proj_sage_sf2.nonmci_cohort_metadata_age_matched` A
INNER JOIN `mining-clinical-decisions.shc_core.demographic` B
ON A.anon_id = B.anon_id

