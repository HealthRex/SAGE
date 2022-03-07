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


select distinct A.proc_id, A.description
from `mining-clinical-decisions.shc_core.order_proc` A
where A.proc_id in (
475
,1369
,1368
,2220
,474
,787
,1435
,735
,2295
,17918
,659308
,1709
,1453
,1036
,2263
,2261
,118571
,17581
,1316
,706
,118573
,118367
,1721
,37118
,26702
,464825
,17968
,118576
,189214
,2265
,34370
,1287
,958
,147416
,445875
,391410
,412485
,1428
,34544
,1717
,68376
,34496
,1434
,400099
,100973
,2471
,1916
,1944
,46821
,1285
,832
,1575
,147420
,34504
,147415
,1883
,703
,1508
,2291
,1001
,999416
,304577
,119815
,464355
,2233
,118575
,1048
,1826
,1634
,2637
,153334
,37225
,198428
,34506
,2215
,418745
,415338
,37113
,17587
,118570
,9994130
,999414
,900
,1590
,2470
,189185
,337
,1215
,1429
,639
,17678
,2349
,153360
,310
,482552
,501391
,999415
,189239
,416542
,35000004791) 
and A.description not like 'HX%'
and A.description not like '%Vital Signs%'
and A.description not like '%PERIPHERAL IV INSERTION CARE%'
and A.description not like '%DISCHARGE%'
and A.description not like '%PATIENT ECG 12-LEAD%'
and A.description not like '%OTHER ORDER SCANNED REPORT%'
and A.description not like '%OTHER PROCEDURE SCANNED REPORT%'
and A.description not like '%POPULATION HEALTH PROTOCOL AUTHORIZATION%'
and A.description not like '%HX OTH ORD SCAN REPORT%'
