/* 
Edited by Sajjad:
Canghed starr_datalake2018 to sch_core_2021
Changed op.jc_uid to op.anon_id
Changed DM.rit_uid to DM.anon_id
Changed enc.jc_uid to enc.anon_id
Basically changed all ic_uid to anon_id
The appointment time formats in shc_core (order_time_jittered, appt_when_jittered) were STRING and I used PARSE_DATETIME to change the dtype to date time.
In the last step I inner joined the cohort with diagnosis table
*/
WITH
  PC AS
  (
  	select DM.* ,op.pat_enc_csn_id_coded as PC_enc, 
	      enc.appt_when_jittered as PC_app_datetime, PARSE_DATETIME("%Y-%m-%d %H:%M:%S",op.order_time_jittered) as PC_ref_datetime,
        (case when DM.gender='Male' then 0 else 1 end) as gender_bool, 
        DATETIME_DIFF( CAST(enc.appt_when_jittered as DATETIME) ,CAST( DM.birth_date_jittered as DATETIME), YEAR) as age
		from `shc_core_2021.shc_order_proc` as op 
		  join `shc_core_2021.shc_encounter` as enc on op.pat_enc_csn_id_coded = enc.pat_enc_csn_id_coded 
      join `shc_core_2021.shc_demographic` as DM on (op.anon_id = DM.anon_id)
		where proc_code = 'REF31' -- REFERRAL TO ENDOCRINE CLINIC (internal)
		and ordering_mode = 'Outpatient'
  ),
  
  SP AS
	(
		select enc.anon_id, enc.pat_enc_csn_id_coded as SP_enc, PARSE_DATETIME("%Y-%m-%d %H:%M:%S",enc.appt_when_jittered) as SP_app_datetime
		from `shc_core_2021.shc_encounter` as enc join `shc_core_2021.shc_dep_map` as dep on enc.department_id = dep.department_id    
    --join `shc_core_2021.diagnosis_code` as DX on (enc.pat_enc_csn_id_coded = DX.pat_enc_csn_id_coded)
		where dep.specialty_dep_c = '7' -- dep.specialty like 'Endocrin%'
    		and visit_type like 'NEW PATIENT%' -- Naturally screens to only 'Office Visit' enc_type 
		-- and appt_type in ('Office Visit','Appointment') -- Otherwise Telephone, Refill, Orders Only, etc.
		and appt_status = 'Completed'
	),
  
  COHORT AS
  (
  SELECT PC.*, SP.* EXCEPT (anon_id)
  FROM PC JOIN SP USING (anon_id) --ON PC.anon_id = SP.anon_id --USING (anon_id)
  WHERE SP.SP_app_datetime BETWEEN PC.PC_ref_datetime AND DATETIME_ADD(PC.PC_ref_datetime, INTERVAL 4 MONTH)
  ORDER BY PC.anon_id 
)
SELECT *
FROM COHORT
INNER JOIN `som-nero-phi-jonc101.shc_core_2021.shc_diagnosis` as diag_t
ON COHORT.anon_id = diag_t.anon_id
ORDER BY COHORT.anon_id 
LIMIT 100000
--SELECT canonical_race ,canonical_ethnicity, gender, count(*) FROM COHORT
--GROUP BY canonical_race ,canonical_ethnicity, gender
--SELECT SUM(case when gender='Female' then 0 else 1 end) AS Male_count
--FROM COHORT