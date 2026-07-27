# NeuroVascNet — Full Data-Quality Audit

**Ruleset:** `1.0.0-frozen-2026-07-25` (frozen before any metric was computed)  
**Recordings in manifest:** 610  
**Recordings with a QC row:** 612  
**Subjects assessed:** 97

Every threshold applied here is declared in `qc/rules.py` with its rationale. No threshold was chosen after seeing a result, and no model performance was consulted at any point in this audit.

## 1. Manifest coverage

| dataset | modality | data_role | in_manifest | qc_rows | complete |
|---|---|---|---|---|---|
| HEFMI | eeg | epoched | 98 | 98 | yes |
| HEFMI | fnirs | epoched_conc | 98 | 100 | yes |
| TUBerlin | eeg | raw | 78 | 78 | yes |
| TUBerlin | fnirs | conc | 78 | 78 | yes |
| ds002336 | eeg | corrected | 53 | 53 | yes |
| ds002336 | eeg | raw | 54 | 54 | yes |
| ds002336 | fmri | raw | 55 | 55 | yes |
| ds002739 | eeg | corrected | 48 | 48 | yes |
| ds002739 | fmri | raw | 48 | 48 | yes |


## 2. Recording-level decisions

| dataset | modality | qc_decision | n |
|---|---|---|---|
| HEFMI | eeg | FAIL | 96 |
| HEFMI | eeg | WARN | 2 |
| HEFMI | fnirs | PASS | 100 |
| TUBerlin | eeg | FAIL | 39 |
| TUBerlin | eeg | PASS | 3 |
| TUBerlin | eeg | WARN | 36 |
| TUBerlin | fnirs | PASS | 52 |
| TUBerlin | fnirs | WARN | 26 |
| ds002336 | eeg | FAIL | 67 |
| ds002336 | eeg | WARN | 40 |
| ds002336 | fmri | FAIL | 1 |
| ds002336 | fmri | PASS | 6 |
| ds002336 | fmri | WARN | 48 |
| ds002739 | eeg | FAIL | 46 |
| ds002739 | eeg | WARN | 2 |
| ds002739 | fmri | PASS | 37 |
| ds002739 | fmri | WARN | 11 |


## 3. Subject-level decisions

| dataset | decision | n |
|---|---|---|
| HEFMI | EXCLUDE | 36 |
| HEFMI | INCLUDE_PARTIAL | 1 |
| TUBerlin | EXCLUDE | 12 |
| TUBerlin | INCLUDE | 12 |
| TUBerlin | INCLUDE_PARTIAL | 2 |
| ds002336 | EXCLUDE | 1 |
| ds002336 | INCLUDE_PARTIAL | 9 |
| ds002739 | EXCLUDE | 23 |
| ds002739 | INCLUDE | 1 |


### Included subjects

| dataset | subject | decision | n_recordings | n_pass | n_warn | n_fail | reason |
|---|---|---|---|---|---|---|---|
| HEFMI | Normals/subject12 | INCLUDE_PARTIAL | 6 | 3 | 2 | 1 | 1/3 eeg recordings excluded |
| TUBerlin | VP001 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP003 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP004 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP005 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP007 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP011 | INCLUDE_PARTIAL | 6 | 2 | 2 | 2 | 2/3 eeg recordings excluded |
| TUBerlin | VP012 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP014 | INCLUDE_PARTIAL | 6 | 2 | 3 | 1 | 1/3 eeg recordings excluded |
| TUBerlin | VP019 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP022 | INCLUDE | 6 | 5 | 1 | 0 | all required modalities have usable recordings |
| TUBerlin | VP023 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP024 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP025 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| TUBerlin | VP026 | INCLUDE | 6 | 2 | 4 | 0 | all required modalities have usable recordings |
| ds002336 | sub-xp102 | INCLUDE_PARTIAL | 16 | 0 | 9 | 7 | 1/5 fmri recordings excluded |
| ds002336 | sub-xp103 | INCLUDE_PARTIAL | 18 | 0 | 11 | 7 | 1/6 eeg recordings excluded |
| ds002336 | sub-xp104 | INCLUDE_PARTIAL | 12 | 0 | 7 | 5 | 1/4 eeg recordings excluded |
| ds002336 | sub-xp105 | INCLUDE_PARTIAL | 18 | 0 | 11 | 7 | 1/6 eeg recordings excluded |
| ds002336 | sub-xp106 | INCLUDE_PARTIAL | 18 | 0 | 11 | 7 | 1/6 eeg recordings excluded |
| ds002336 | sub-xp107 | INCLUDE_PARTIAL | 18 | 0 | 10 | 8 | 2/6 eeg recordings excluded |
| ds002336 | sub-xp108 | INCLUDE_PARTIAL | 18 | 2 | 9 | 7 | 1/6 eeg recordings excluded |
| ds002336 | sub-xp109 | INCLUDE_PARTIAL | 14 | 0 | 9 | 5 | 1/4 eeg recordings excluded |
| ds002336 | sub-xp110 | INCLUDE_PARTIAL | 18 | 4 | 7 | 7 | 1/6 eeg recordings excluded |
| ds002739 | sub-23 | INCLUDE | 4 | 2 | 2 | 0 | all required modalities have usable recordings |


### Excluded subjects, with documented reason

| dataset | subject | reason | failed_recording_reasons |
|---|---|---|---|
| HEFMI | Normals/subject1 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1272.2866 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1077.1711 (limit |
| HEFMI | Normals/subject10 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=2957.0099 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=3177.4027 (limit |
| HEFMI | Normals/subject11 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=811.7648 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1464.1014 (limit  |
| HEFMI | Normals/subject13 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=2668.6711 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1736.2549 (limit |
| HEFMI | Normals/subject14 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=468.1875 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=602.0674 (limit 1 |
| HEFMI | Normals/subject15 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=264.1704 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=482.0334 (limit 1 |
| HEFMI | Normals/subject16 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1767.8454 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=2312.9314 (limit |
| HEFMI | Normals/subject17 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1086.4083 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1309.4139 (limit |
| HEFMI | Normals/subject2 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=931.3266 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=685.9095 (limit 1 |
| HEFMI | Normals/subject3 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1173.9639 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=155.3205 (limit  |
| HEFMI | Normals/subject4 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=593.5508 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1357.5436 (limit  |
| HEFMI | Normals/subject5 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1394.7146 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=2803.8425 (limit |
| HEFMI | Normals/subject6 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1380.3072 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=487.9463 (limit  |
| HEFMI | Normals/subject7 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] robust_amplitude_uv=14259.3669 (limit 5000.0): implausible amplitude for scalp EEG in uV; line_noise_ratio=252.219 (limit 100.0): mains contamination re |
| HEFMI | Normals/subject8 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1291.0744 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1227.5547 (limit |
| HEFMI | Normals/subject9 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1338.8675 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=2886.694 (limit  |
| HEFMI | Patients/subject1 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=215.0516 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=166.7097 (limit 1 |
| HEFMI | Patients/subject10 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=2191.0067 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=2575.7174 (limit |
| HEFMI | Patients/subject11 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=268.3406 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=673.7239 (limit 1 |
| HEFMI | Patients/subject12 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=975.5018 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1249.8064 (limit  |
| HEFMI | Patients/subject13 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=14715.4309 (limit 100.0): mains contamination relative to neighbourhood; transients_per_min_per_ch=89.1004 (limit 60.0): high-amplitude |
| HEFMI | Patients/subject14 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1810.6817 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1729.8897 (limit |
| HEFMI | Patients/subject15 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=298.8181 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=535.0213 (limit 1 |
| HEFMI | Patients/subject16 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=515.5764 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=864.0163 (limit 1 |
| HEFMI | Patients/subject17 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=476.6471 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1945.3065 (limit  |
| HEFMI | Patients/subject18 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1188.2505 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=693.6057 (limit  |
| HEFMI | Patients/subject19 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=2973.4685 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=1691.2206 (limit |
| HEFMI | Patients/subject2 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=1152.407 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=17389.9312 (limit |
| HEFMI | Patients/subject20 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=539.6851 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=948.4582 (limit 1 |
| HEFMI | Patients/subject3 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=527.2368 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=553.4026 (limit 1 |
| HEFMI | Patients/subject4 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] robust_amplitude_uv=11007.8555 (limit 5000.0): implausible amplitude for scalp EEG in uV || [eeg/motor_imagery/epoched] robust_amplitude_uv=10999.0957 ( |
| HEFMI | Patients/subject5 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=470.6097 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=547.2201 (limit 1 |
| HEFMI | Patients/subject6 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] robust_amplitude_uv=12657.6929 (limit 5000.0): implausible amplitude for scalp EEG in uV || [eeg/motor_imagery/epoched] robust_amplitude_uv=12540.1611 ( |
| HEFMI | Patients/subject7 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=343.0208 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=290.1046 (limit 1 |
| HEFMI | Patients/subject8 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=500.2884 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject9 | all eeg recordings failed QC | [eeg/motor_imagery/epoched] line_noise_ratio=3109.494 (limit 100.0): mains contamination relative to neighbourhood || [eeg/motor_imagery/epoched] line_noise_ratio=3154.2236 (limit  |
| TUBerlin | VP002 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=431.1014 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=428.8756 (limit 100.0): mains contamination |
| TUBerlin | VP006 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=197.1471 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=344.5041 (limit 100.0): mains contamination |
| TUBerlin | VP008 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=199.8494 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=371.0892 (limit 100.0): mains contamination |
| TUBerlin | VP009 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=172.2712 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=285.869 (limit 100.0): mains contamination  |
| TUBerlin | VP010 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=162.8807 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=1998.9687 (limit 100.0): mains contaminatio |
| TUBerlin | VP013 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=1311.4694 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=1353.9212 (limit 100.0): mains contaminati |
| TUBerlin | VP015 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=103.1194 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=168.6244 (limit 100.0): mains contamination |
| TUBerlin | VP016 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=158.3656 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=437.6843 (limit 100.0): mains contamination |
| TUBerlin | VP017 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=713.9731 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=363.5096 (limit 100.0): mains contamination |
| TUBerlin | VP018 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=198.6284 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=576.6608 (limit 100.0): mains contamination |
| TUBerlin | VP020 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=105.653 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=292.3485 (limit 100.0): mains contamination  |
| TUBerlin | VP021 | all eeg recordings failed QC | [eeg/dsr/raw] line_noise_ratio=192.6405 (limit 100.0): mains contamination relative to neighbourhood || [eeg/nback/raw] line_noise_ratio=461.2056 (limit 100.0): mains contamination |
| ds002336 | sub-xp101 | all eeg recordings failed QC | [eeg/eegfmriNF/corrected] corrected_robust_amplitude_uv=743.5005 (limit 500.0): corrected trace still does not look like scalp EEG || [eeg/eegNF/corrected] corrected_robust_amplitu |
| ds002739 | sub-01 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=64.338 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmonic |
| ds002739 | sub-02 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=58.7089 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-03 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=58.7089 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-04 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=120.2669 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmon |
| ds002739 | sub-05 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=188.5558 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmon |
| ds002739 | sub-06 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=89.2366 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-07 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=76.7062 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-08 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] bad_channel_fraction=0.6094 (limit 0.33): too many flat/railed/uncorrelated channels to interpolate; gradient_harmonic_ratio_max=100.5692 (limit 10.0): |
| ds002739 | sub-09 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=98.8609 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-10 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=110.2049 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmon |
| ds002739 | sub-11 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=68.1612 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-12 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=110.1004 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmon |
| ds002739 | sub-13 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=96.5346 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-14 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=88.3281 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-15 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=116.1405 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmon |
| ds002739 | sub-16 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=123.5237 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmon |
| ds002739 | sub-17 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=55.1255 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-18 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=99.5863 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-19 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=106.6123 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmon |
| ds002739 | sub-20 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=117.5242 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmon |
| ds002739 | sub-21 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=46.0316 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-22 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=51.9056 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmoni |
| ds002739 | sub-24 | all eeg recordings failed QC | [eeg/rdm_decision/corrected] gradient_harmonic_ratio_max=113.7689 (limit 10.0): residual scanner-gradient harmonics after correction || [eeg/rdm_decision/corrected] gradient_harmon |


## 4. Every failed / errored recording

| dataset | subject | session | task | modality | data_role | qc_decision | qc_reason |
|---|---|---|---|---|---|---|---|
| HEFMI | Normals/subject1 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1272.2866 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject1 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1077.1711 (limit 100.0): mains contamination relative to neighbourhood |
| ds002336 | sub-xp101 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=9084.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=9084.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp101 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=7653.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=7653.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp101 | ses-01 | eegfmriNF | eeg | corrected | FAIL | corrected_robust_amplitude_uv=743.5005 (limit 500.0): corrected trace still does not look like scalp EEG |
| ds002336 | sub-xp101 | ses-01 | eegNF | eeg | corrected | FAIL | corrected_robust_amplitude_uv=598.9 (limit 500.0): corrected trace still does not look like scalp EEG |
| ds002739 | sub-01 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=64.338 (limit 10.0): residual scanner-gradient harmonics after correction |
| HEFMI | Normals/subject10 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2957.0099 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject10 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=3177.4027 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject10 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2460.4441 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject11 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=811.7648 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject11 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1464.1014 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject11 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1001.0693 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject12 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=139.023 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject13 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2668.6711 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject13 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1736.2549 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject13 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2675.3461 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject14 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=468.1875 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject14 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=602.0674 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject14 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=550.0922 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject15 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=264.1704 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject15 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=482.0334 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject15 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=444.659 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject16 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1767.8454 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject16 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2312.9314 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject16 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=903.2004 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject17 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1086.4083 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject17 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1309.4139 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject17 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=4612.0321 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject2 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=931.3266 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject2 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=685.9095 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject2 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=480.4517 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject3 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1173.9639 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject3 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=155.3205 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject3 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=185.7562 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject4 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=593.5508 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject4 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1357.5436 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject4 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=3370.0461 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject5 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1394.7146 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject5 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2803.8425 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject5 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1510.5832 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject6 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1380.3072 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject6 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=487.9463 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject6 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=766.7393 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject7 | 1 | motor_imagery | eeg | epoched | FAIL | robust_amplitude_uv=14259.3669 (limit 5000.0): implausible amplitude for scalp EEG in uV; line_noise_ratio=252.219 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject7 | 2 | motor_imagery | eeg | epoched | FAIL | robust_amplitude_uv=15496.7339 (limit 5000.0): implausible amplitude for scalp EEG in uV; line_noise_ratio=2783.6568 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject7 | 3 | motor_imagery | eeg | epoched | FAIL | robust_amplitude_uv=15708.2163 (limit 5000.0): implausible amplitude for scalp EEG in uV; line_noise_ratio=3449.7246 (limit 100.0): mains contamination relative to neighbourhood; t |
| HEFMI | Normals/subject8 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1291.0744 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject8 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1227.5547 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject8 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=912.9984 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject9 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1338.8675 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject9 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2886.694 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Normals/subject9 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1177.4562 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject1 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=215.0516 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject1 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=166.7097 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject1 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1938.4813 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject10 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2191.0067 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject10 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2575.7174 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject10 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1699.8372 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject11 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=268.3406 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject11 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=673.7239 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject12 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=975.5018 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject12 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1249.8064 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject12 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1088.991 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject13 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=14715.4309 (limit 100.0): mains contamination relative to neighbourhood; transients_per_min_per_ch=89.1004 (limit 60.0): high-amplitude transient rate indicates mo |
| HEFMI | Patients/subject13 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=467.8684 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject13 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1100.4036 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject14 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1810.6817 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject14 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1729.8897 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject14 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1892.7161 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject15 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=298.8181 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject15 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=535.0213 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject16 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=515.5764 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject16 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=864.0163 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject17 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=476.6471 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject17 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1945.3065 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject18 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1188.2505 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject18 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=693.6057 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject19 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=2973.4685 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject19 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1691.2206 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject2 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=1152.407 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject2 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=17389.9312 (limit 100.0): mains contamination relative to neighbourhood; transients_per_min_per_ch=132.724 (limit 60.0): high-amplitude transient rate indicates mo |
| HEFMI | Patients/subject2 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=5901.8053 (limit 100.0): mains contamination relative to neighbourhood; transients_per_min_per_ch=100.6064 (limit 60.0): high-amplitude transient rate indicates mo |
| HEFMI | Patients/subject20 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=539.6851 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject20 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=948.4582 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject3 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=527.2368 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject3 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=553.4026 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject3 | 3 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=589.3436 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject4 | 1 | motor_imagery | eeg | epoched | FAIL | robust_amplitude_uv=11007.8555 (limit 5000.0): implausible amplitude for scalp EEG in uV |
| HEFMI | Patients/subject4 | 2 | motor_imagery | eeg | epoched | FAIL | robust_amplitude_uv=10999.0957 (limit 5000.0): implausible amplitude for scalp EEG in uV |
| HEFMI | Patients/subject4 | 3 | motor_imagery | eeg | epoched | FAIL | robust_amplitude_uv=11017.7543 (limit 5000.0): implausible amplitude for scalp EEG in uV; line_noise_ratio=102.7066 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject5 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=470.6097 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject5 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=547.2201 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject5 | 3 | motor_imagery | eeg | epoched | FAIL | robust_amplitude_uv=10620.1312 (limit 5000.0): implausible amplitude for scalp EEG in uV; line_noise_ratio=29947.4129 (limit 100.0): mains contamination relative to neighbourhood;  |
| HEFMI | Patients/subject6 | 1 | motor_imagery | eeg | epoched | FAIL | robust_amplitude_uv=12657.6929 (limit 5000.0): implausible amplitude for scalp EEG in uV |
| HEFMI | Patients/subject6 | 2 | motor_imagery | eeg | epoched | FAIL | robust_amplitude_uv=12540.1611 (limit 5000.0): implausible amplitude for scalp EEG in uV; transients_per_min_per_ch=88.8352 (limit 60.0): high-amplitude transient rate indicates mo |
| HEFMI | Patients/subject7 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=343.0208 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject7 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=290.1046 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject8 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=500.2884 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject9 | 1 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=3109.494 (limit 100.0): mains contamination relative to neighbourhood |
| HEFMI | Patients/subject9 | 2 | motor_imagery | eeg | epoched | FAIL | line_noise_ratio=3154.2236 (limit 100.0): mains contamination relative to neighbourhood |
| ds002336 | sub-xp101 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=7777.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=7777.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp101 | ses-01 | motorloc | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8150.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8150.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp101 | ses-01 | fmriNF | eeg | corrected | FAIL | corrected_robust_amplitude_uv=948.6 (limit 500.0): corrected trace still does not look like scalp EEG |
| ds002336 | sub-xp101 | ses-01 | motorloc | eeg | corrected | FAIL | gradient_harmonic_ratio_max=81.0494 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp102 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8279.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8279.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp102 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8215.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8215.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp102 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8389.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8389.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp102 | ses-01 | MIpost | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8104.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8104.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp102 | ses-01 | MIpre | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=7787.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=7787.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp102 | ses-01 | motorloc | eeg | raw | FAIL | robust_amplitude_uv_low=0.0 (limit 0.05): signal is effectively flat; clipped_fraction=1.0 (limit 0.01): samples pinned at the ADC rail; constant_fraction=1.0 (limit 0.05): extende |
| ds002336 | sub-xp103 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8001.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8001.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp103 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8452.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8452.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp103 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8401.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8401.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp103 | ses-01 | MIpost | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8285.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8285.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp103 | ses-01 | MIpre | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8155.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8155.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp103 | ses-01 | motorloc | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8152.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8152.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp103 | ses-01 | motorloc | eeg | corrected | FAIL | gradient_harmonic_ratio_max=45.889 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp104 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8494.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8494.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp104 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8316.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8316.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp104 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8440.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8440.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp104 | ses-01 | motorloc | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=5196.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=5196.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp104 | ses-01 | motorloc | eeg | corrected | FAIL | gradient_harmonic_ratio_max=13.6708 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp105 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=7988.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=7988.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp105 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8463.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8463.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp105 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8321.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8321.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp105 | ses-01 | MIpost | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8243.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8243.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp105 | ses-01 | MIpre | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8291.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8291.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp105 | ses-01 | motorloc | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8278.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8278.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp105 | ses-01 | motorloc | eeg | corrected | FAIL | gradient_harmonic_ratio_max=107.1415 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp106 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8308.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8308.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp106 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8258.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8258.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp106 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8225.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8225.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp106 | ses-01 | MIpost | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=13169.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=13169.5 (limit 5000.0): implausible |
| ds002336 | sub-xp106 | ses-01 | MIpre | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=6957.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=6957.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp106 | ses-01 | motorloc | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8329.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8329.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp106 | ses-01 | motorloc | eeg | corrected | FAIL | gradient_harmonic_ratio_max=64.6156 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp107 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=9097.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=9097.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp107 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8615.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8615.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp107 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8951.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8951.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp107 | ses-01 | MIpost | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=9199.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=9199.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp107 | ses-01 | MIpre | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8307.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8307.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp107 | ses-01 | motorloc | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8400.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8400.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp107 | ses-01 | eegNF | eeg | corrected | FAIL | gradient_harmonic_ratio_max=24.5097 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp107 | ses-01 | motorloc | eeg | corrected | FAIL | gradient_harmonic_ratio_max=94.4467 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp108 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=9030.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=9030.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp108 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8837.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8837.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp108 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=9043.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=9043.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp108 | ses-01 | MIpost | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8989.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8989.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp108 | ses-01 | MIpre | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=9111.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=9111.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp108 | ses-01 | motorloc | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=9057.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=9057.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp108 | ses-01 | motorloc | eeg | corrected | FAIL | gradient_harmonic_ratio_max=95.451 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp109 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8109.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8109.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp109 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8099.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8099.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp109 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8093.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8093.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp109 | ses-01 | motorloc | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=7406.0 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=7406.0 (limit 5000.0): implausible a |
| ds002336 | sub-xp109 | ses-01 | motorloc | eeg | corrected | FAIL | gradient_harmonic_ratio_max=43.2029 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp110 | ses-01 | eegfmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8413.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8413.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp110 | ses-01 | eegNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8380.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8380.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp110 | ses-01 | fmriNF | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8430.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8430.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp110 | ses-01 | MIpost | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8390.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8390.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp110 | ses-01 | MIpre | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8370.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8370.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp110 | ses-01 | motorloc | eeg | raw | FAIL | raw_amplitude_hard_limit_uv=8267.5 (limit 5000.0): uncorrected in-scanner EEG above the 5 mV limit must not enter training; robust_amplitude_uv=8267.5 (limit 5000.0): implausible a |
| ds002336 | sub-xp110 | ses-01 | motorloc | eeg | corrected | FAIL | gradient_harmonic_ratio_max=51.3124 (limit 10.0): residual scanner-gradient harmonics after correction |
| TUBerlin | VP002 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=431.1014 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP002 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=428.8756 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP002 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=299.4229 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP006 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=197.1471 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP006 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=344.5041 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP006 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=462.2803 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP008 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=199.8494 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP008 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=371.0892 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP008 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=146.6561 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP009 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=172.2712 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP009 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=285.869 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP009 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=162.0761 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP010 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=162.8807 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP010 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=1998.9687 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP010 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=174.7242 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP011 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=468.6962 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP011 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=159.502 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP013 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=1311.4694 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP013 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=1353.9212 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP013 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=788.7743 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP014 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=115.8628 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP015 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=103.1194 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP015 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=168.6244 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP015 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=134.9007 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP016 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=158.3656 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP016 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=437.6843 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP016 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=232.701 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP017 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=713.9731 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP017 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=363.5096 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP017 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=230.3127 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP018 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=198.6284 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP018 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=576.6608 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP018 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=215.8653 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP020 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=105.653 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP020 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=292.3485 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP020 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=175.2101 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP021 | ses-01 | dsr | eeg | raw | FAIL | line_noise_ratio=192.6405 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP021 | ses-01 | nback | eeg | raw | FAIL | line_noise_ratio=461.2056 (limit 100.0): mains contamination relative to neighbourhood |
| TUBerlin | VP021 | ses-01 | wg | eeg | raw | FAIL | line_noise_ratio=350.1974 (limit 100.0): mains contamination relative to neighbourhood |
| ds002739 | sub-01 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=80.5049 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-02 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=58.7089 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-02 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=85.3533 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-03 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=58.7089 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-03 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=85.3533 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-04 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=120.2669 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-04 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=96.8368 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-05 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=188.5558 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-05 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=159.4583 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-06 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=89.2366 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-06 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=80.1383 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-07 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=76.7062 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-07 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=70.2389 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-08 | run-01 | rdm_decision | eeg | corrected | FAIL | bad_channel_fraction=0.6094 (limit 0.33): too many flat/railed/uncorrelated channels to interpolate; gradient_harmonic_ratio_max=100.5692 (limit 10.0): residual scanner-gradient ha |
| ds002739 | sub-08 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=43.36 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-09 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=98.8609 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-09 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=75.201 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-10 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=110.2049 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-10 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=148.8389 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-11 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=68.1612 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-11 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=64.7841 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-12 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=110.1004 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-12 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=91.6906 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-13 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=96.5346 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-13 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=75.1964 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-14 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=88.3281 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-14 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=105.9924 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-15 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=116.1405 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-15 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=103.4122 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-16 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=123.5237 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-16 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=42.3364 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-17 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=55.1255 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-17 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=67.8444 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-18 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=99.5863 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-18 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=105.1341 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-19 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=106.6123 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-19 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=100.1629 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-20 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=117.5242 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-20 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=109.0429 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-21 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=46.0316 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-21 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=63.5473 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-22 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=51.9056 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-22 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=52.6255 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-24 | run-01 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=113.7689 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002739 | sub-24 | run-02 | rdm_decision | eeg | corrected | FAIL | gradient_harmonic_ratio_max=140.1782 (limit 10.0): residual scanner-gradient harmonics after correction |
| ds002336 | sub-xp102 | ses-01 | MIpre | fmri | raw | FAIL | mean_fd_mm=0.5776 (limit 0.5): excessive head motion |


## 5. Modality summaries


### EEG

| dataset | fs_hz_median | fs_hz_min | fs_hz_max | n_channels_median | n_channels_min | n_channels_max | duration_s_median | duration_s_min | duration_s_max | robust_amplitude_uv_median | robust_amplitude_uv_min | robust_amplitude_uv_max | nan_fraction_median | nan_fraction_min | nan_fraction_max | clipped_fraction_median | clipped_fraction_min | clipped_fraction_max | line_noise_ratio_median | line_noise_ratio_min | line_noise_ratio_max | bad_channel_fraction_median | bad_channel_fraction_min | bad_channel_fraction_max | transients_per_min_per_ch_median | transients_per_min_per_ch_min | transients_per_min_per_ch_max | gradient_harmonic_ratio_max_median | gradient_harmonic_ratio_max_min | gradient_harmonic_ratio_max_max | bcg_residual_ratio_median | bcg_residual_ratio_min | bcg_residual_ratio_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HEFMI | 256 | 256 | 256 | 32 | 32 | 32 | 1117 | 1117 | 1117 | 127.2 | 58.01 | 1.571e+04 | 0 | 0 | 0 | 3.279e-07 | 2.186e-07 | 4.372e-07 | 1039 | 31.53 | 2.995e+04 | 0 | 0 | 0.2188 | 6.332 | 0 | 132.7 |  |  |  |  |  |  |
| TUBerlin | 200 | 200 | 200 | 30 | 30 | 30 | 1860 | 1313 | 1976 | 326.2 | 113.4 | 1519 | 0 | 0 | 0 | 1.793e-07 | 1.687e-07 | 2.54e-07 | 99.35 | 3.361 | 1999 | 0.06667 | 0 | 0.2333 | 9.409 | 1.721 | 44.2 |  |  |  |  |  |  |
| ds002336 | 5000 | 200 | 5000 | 64 | 63 | 64 | 401 | 202 | 1101 | 948.6 | 0 | 1.317e+04 | 0 | 0 | 0 | 0.0002305 | 1.442e-07 | 1 | 0.07546 | 0.0008859 | 0.9983 | 0.01587 | 0 | 1 | 0 | 0 | 35.87 | 526.2 | 0.9939 | 9200 | 0.3042 | 0.07288 | 1.811 |
| ds002739 | 1000 | 1000 | 1000 | 64 | 58 | 64 | 1602 | 1590 | 1620 | 39.1 | 14.55 | 124.7 | 0 | 0 | 0 | 1.953e-08 | 1.94e-08 | 2.152e-08 | 0.1101 | 0.04796 | 0.657 | 0.04068 | 0 | 0.6094 | 3.773 | 0.001757 | 22.22 | 88.78 | 5.498 | 188.6 | 0.2921 | 0.1044 | 0.8823 |


### FNIRS

| dataset | fs_hz_median | fs_hz_min | fs_hz_max | n_channels_median | n_channels_min | n_channels_max | channel_availability_median | channel_availability_min | channel_availability_max | poor_coupling_fraction_median | poor_coupling_fraction_min | poor_coupling_fraction_max | saturated_fraction_median | saturated_fraction_min | saturated_fraction_max | flatline_fraction_median | flatline_fraction_min | flatline_fraction_max | motion_per_min_per_ch_median | motion_per_min_per_ch_min | motion_per_min_per_ch_max | max_abs_conc_umol_median | max_abs_conc_umol_min | max_abs_conc_umol_max | baseline_sd_umol_median | baseline_sd_umol_min | baseline_sd_umol_max | n_usable_trials_median | n_usable_trials_min | n_usable_trials_max | gsr_variance_removed_frac_median | gsr_variance_removed_frac_min | gsr_variance_removed_frac_max | gsr_evoked_amplitude_retained_frac_median | gsr_evoked_amplitude_retained_frac_min | gsr_evoked_amplitude_retained_frac_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HEFMI | 11 | 11 | 11 | 90 | 90 | 90 | 1 | 1 | 1 | 0 | 0 | 0 | 2.717e-06 | 1.811e-06 | 5.433e-06 | 0 | 0 | 0 | 0.07291 | 0 | 1.467 | 1.251e-05 | 3.428e-06 | 4.785e-05 | 7.96e-08 | 2.369e-08 | 1.759e-07 | 30 | 30 | 30 | 0.1814 | 0.038 | 0.7399 |  |  |  |
| TUBerlin | 10 | 10 | 10 | 36 | 36 | 36 | 1 | 1 | 1 | 0 | 0 | 0.02778 | 2.749e-06 | 2.618e-06 | 3.848e-06 | 0 | 0 | 0 | 0.1131 | 0.01401 | 0.478 | 0.03273 | 0.01346 | 1.212 | 0.0009435 | 0.0004372 | 0.003482 | 27 | 17 | 60 | 0.5247 | 0.1988 | 0.9826 | 0.1717 | 0.01706 | 216.9 |


### FMRI

| dataset | n_volumes_median | n_volumes_min | n_volumes_max | tr_s_median | tr_s_min | tr_s_max | median_tsnr_median | median_tsnr_min | median_tsnr_max | mean_fd_mm_median | mean_fd_mm_min | mean_fd_mm_max | censored_fraction_median | censored_fraction_min | censored_fraction_max | n_usable_volumes_median | n_usable_volumes_min | n_usable_volumes_max | roi_coverage_median | roi_coverage_min | roi_coverage_max | n_events_median | n_events_min | n_events_max | motor_roi_evoked_peak_median | motor_roi_evoked_peak_min | motor_roi_evoked_peak_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ds002336 | 210 | 55 | 210 | 2 | 2 | 2 | 23.89 | 10.13 | 35.51 | 0.1241 | 0.06726 | 0.5776 | 0.004762 | 0 | 0.2222 | 182 | 84 | 210 | 1 | 0.9583 | 1 | 20 | 10 | 20 | 0.0009965 | -0.2931 | 0.3415 |
| ds002739 | 771 | 771 | 771 | 2 | 2 | 2 | 189.8 | 117.3 | 266.1 |  |  |  | 0.01038 | 0 | 0.09598 | 763 | 697 | 771 | 0.9792 | 0.8542 | 0.9792 | 6202 | 6054 | 6276 | 0.002205 | -0.02011 | 0.02093 |


## 6. Simultaneous EEG-fMRI: raw vs corrected

| subject | task | raw_robust_amplitude_uv | corrected_robust_amplitude_uv | power_reduction_db | raw_gradient_harmonic_ratio_max | corrected_gradient_harmonic_ratio_max | correction_effective |
|---|---|---|---|---|---|---|---|
| sub-xp101 | eegfmriNF | 9084 | 743.5 | 19.19 | 3256 | 3.509 | 1 |
| sub-xp101 | eegNF | 7654 | 598.9 | 21.39 | 2498 | 1.924 | 1 |
| sub-xp101 | fmriNF | 7777 | 948.6 | 25.64 | 3958 | 1.246 | 1 |
| sub-xp101 | motorloc | 8150 | 413.5 | 25.16 | 5752 | 81.05 | 1 |
| sub-xp102 | eegfmriNF | 8280 | 141.4 | 35.77 | 7979 | 1.835 | 1 |
| sub-xp102 | eegNF | 8215 | 118.4 | 37.14 | 5789 | 1.557 | 1 |
| sub-xp102 | fmriNF | 8390 | 180.9 | 31.49 | 6646 | 1.353 | 1 |
| sub-xp102 | MIpost | 8104 | 147.1 | 37.98 | 4095 | 1.457 | 1 |
| sub-xp102 | MIpre | 7787 | 386.5 | 27.19 | 3866 | 2.65 | 1 |
| sub-xp103 | eegfmriNF | 8001 | 189.5 | 32.43 | 3809 | 1.122 | 1 |
| sub-xp103 | eegNF | 8452 | 159.8 | 33.23 | 7556 | 1.001 | 1 |
| sub-xp103 | fmriNF | 8402 | 166.4 | 32.83 | 6659 | 0.9984 | 1 |
| sub-xp103 | MIpost | 8286 | 258.7 | 32.16 | 5745 | 1.013 | 1 |
| sub-xp103 | MIpre | 8156 | 178.8 | 31.29 | 4364 | 1.001 | 1 |
| sub-xp103 | motorloc | 8152 | 168.2 | 29.92 | 3702 | 45.89 | 1 |
| sub-xp104 | eegfmriNF | 8494 | 179.4 | 32.48 | 9200 | 2.209 | 1 |
| sub-xp104 | eegNF | 8316 | 195.6 | 29.4 | 4285 | 3.132 | 1 |
| sub-xp104 | fmriNF | 8440 | 142.2 | 31.35 | 7632 | 3.539 | 1 |
| sub-xp104 | motorloc | 5196 | 186.9 | 24.59 | 2713 | 13.67 | 1 |
| sub-xp105 | eegfmriNF | 7988 | 207.6 | 28.05 | 5698 | 4.156 | 1 |
| sub-xp105 | eegNF | 8463 | 149 | 30.25 | 4151 | 4.567 | 1 |
| sub-xp105 | fmriNF | 8321 | 149 | 30.7 | 4838 | 2.394 | 1 |
| sub-xp105 | MIpost | 8244 | 150.9 | 31.87 | 5079 | 1.534 | 1 |
| sub-xp105 | MIpre | 8292 | 170 | 30.59 | 3364 | 3.099 | 1 |
| sub-xp105 | motorloc | 8278 | 193.9 | 27.14 | 6409 | 107.1 | 1 |
| sub-xp106 | eegfmriNF | 8308 | 184.7 | 30.22 | 6893 | 2.141 | 1 |
| sub-xp106 | eegNF | 8258 | 169.8 | 33.33 | 4527 | 3.004 | 1 |
| sub-xp106 | fmriNF | 8226 | 162.4 | 32.5 | 5191 | 1.937 | 1 |
| sub-xp106 | MIpost | 1.317e+04 | 143.5 | 34.75 | 945.2 | 2.322 | 1 |
| sub-xp106 | MIpre | 6957 | 185.7 | 31.61 | 3534 | 2.014 | 1 |
| sub-xp106 | motorloc | 8330 | 178.9 | 32.27 | 3638 | 64.62 | 1 |
| sub-xp107 | eegfmriNF | 9097 | 181.7 | 29.81 | 6344 | 1.541 | 1 |
| sub-xp107 | eegNF | 8616 | 235.7 | 29.33 | 5791 | 24.51 | 1 |
| sub-xp107 | fmriNF | 8952 | 175.8 | 30.76 | 7303 | 1.343 | 1 |
| sub-xp107 | MIpost | 9200 | 167.7 | 30.61 | 5417 | 1.199 | 1 |
| sub-xp107 | MIpre | 8307 | 204.2 | 28.98 | 3931 | 1.011 | 1 |
| sub-xp107 | motorloc | 8400 | 174.2 | 28.6 | 3974 | 94.45 | 1 |
| sub-xp108 | eegfmriNF | 9030 | 166.9 | 30.69 | 7517 | 1.081 | 1 |
| sub-xp108 | eegNF | 8838 | 163.9 | 31.27 | 3787 | 1.221 | 1 |
| sub-xp108 | fmriNF | 9044 | 159.7 | 31.57 | 4424 | 1.08 | 1 |
| sub-xp108 | MIpost | 8989 | 142.6 | 32.33 | 6352 | 1.015 | 1 |
| sub-xp108 | MIpre | 9112 | 150.4 | 31.48 | 8205 | 0.9939 | 1 |
| sub-xp108 | motorloc | 9057 | 203.4 | 28.22 | 6773 | 95.45 | 1 |
| sub-xp109 | eegfmriNF | 8109 | 127.4 | 32.35 | 6780 | 1.055 | 1 |
| sub-xp109 | eegNF | 8099 | 127.3 | 33.49 | 4923 | 1.075 | 1 |
| sub-xp109 | fmriNF | 8093 | 133.9 | 32.85 | 7935 | 1.169 | 1 |
| sub-xp109 | motorloc | 7406 | 177.7 | 25.41 | 4179 | 43.2 | 1 |
| sub-xp110 | eegfmriNF | 8414 | 173.9 | 31.79 | 7284 | 1.405 | 1 |
| sub-xp110 | eegNF | 8380 | 190.8 | 31.7 | 3691 | 1.723 | 1 |
| sub-xp110 | fmriNF | 8430 | 232.2 | 32.38 | 3901 | 2.704 | 1 |
| sub-xp110 | MIpost | 8390 | 151.7 | 33 | 3072 | 1.601 | 1 |
| sub-xp110 | MIpre | 8370 | 185 | 30.97 | 3890 | 1.583 | 1 |
| sub-xp110 | motorloc | 8267 | 171.3 | 28.29 | 4642 | 51.31 | 1 |
| sub-xp101 | eegfmriNF | 9084 | 743.5 | 19.19 | 3256 | 3.509 | 1 |
| sub-xp101 | eegNF | 7654 | 598.9 | 21.39 | 2498 | 1.924 | 1 |
| sub-xp101 | fmriNF | 7777 | 948.6 | 25.64 | 3958 | 1.246 | 1 |
| sub-xp101 | motorloc | 8150 | 413.5 | 25.16 | 5752 | 81.05 | 1 |
| sub-xp102 | eegfmriNF | 8280 | 141.4 | 35.77 | 7979 | 1.835 | 1 |
| sub-xp102 | eegNF | 8215 | 118.4 | 37.14 | 5789 | 1.557 | 1 |
| sub-xp102 | fmriNF | 8390 | 180.9 | 31.49 | 6646 | 1.353 | 1 |
| sub-xp102 | MIpost | 8104 | 147.1 | 37.98 | 4095 | 1.457 | 1 |
| sub-xp102 | MIpre | 7787 | 386.5 | 27.19 | 3866 | 2.65 | 1 |
| sub-xp103 | eegfmriNF | 8001 | 189.5 | 32.43 | 3809 | 1.122 | 1 |
| sub-xp103 | eegNF | 8452 | 159.8 | 33.23 | 7556 | 1.001 | 1 |
| sub-xp103 | fmriNF | 8402 | 166.4 | 32.83 | 6659 | 0.9984 | 1 |
| sub-xp103 | MIpost | 8286 | 258.7 | 32.16 | 5745 | 1.013 | 1 |
| sub-xp103 | MIpre | 8156 | 178.8 | 31.29 | 4364 | 1.001 | 1 |
| sub-xp103 | motorloc | 8152 | 168.2 | 29.92 | 3702 | 45.89 | 1 |
| sub-xp104 | eegfmriNF | 8494 | 179.4 | 32.48 | 9200 | 2.209 | 1 |
| sub-xp104 | eegNF | 8316 | 195.6 | 29.4 | 4285 | 3.132 | 1 |
| sub-xp104 | fmriNF | 8440 | 142.2 | 31.35 | 7632 | 3.539 | 1 |
| sub-xp104 | motorloc | 5196 | 186.9 | 24.59 | 2713 | 13.67 | 1 |
| sub-xp105 | eegfmriNF | 7988 | 207.6 | 28.05 | 5698 | 4.156 | 1 |
| sub-xp105 | eegNF | 8463 | 149 | 30.25 | 4151 | 4.567 | 1 |
| sub-xp105 | fmriNF | 8321 | 149 | 30.7 | 4838 | 2.394 | 1 |
| sub-xp105 | MIpost | 8244 | 150.9 | 31.87 | 5079 | 1.534 | 1 |
| sub-xp105 | MIpre | 8292 | 170 | 30.59 | 3364 | 3.099 | 1 |
| sub-xp105 | motorloc | 8278 | 193.9 | 27.14 | 6409 | 107.1 | 1 |
| sub-xp106 | eegfmriNF | 8308 | 184.7 | 30.22 | 6893 | 2.141 | 1 |
| sub-xp106 | eegNF | 8258 | 169.8 | 33.33 | 4527 | 3.004 | 1 |
| sub-xp106 | fmriNF | 8226 | 162.4 | 32.5 | 5191 | 1.937 | 1 |
| sub-xp106 | MIpost | 1.317e+04 | 143.5 | 34.75 | 945.2 | 2.322 | 1 |
| sub-xp106 | MIpre | 6957 | 185.7 | 31.61 | 3534 | 2.014 | 1 |
| sub-xp106 | motorloc | 8330 | 178.9 | 32.27 | 3638 | 64.62 | 1 |
| sub-xp107 | eegfmriNF | 9097 | 181.7 | 29.81 | 6344 | 1.541 | 1 |
| sub-xp107 | eegNF | 8616 | 235.7 | 29.33 | 5791 | 24.51 | 1 |
| sub-xp107 | fmriNF | 8952 | 175.8 | 30.76 | 7303 | 1.343 | 1 |
| sub-xp107 | MIpost | 9200 | 167.7 | 30.61 | 5417 | 1.199 | 1 |
| sub-xp107 | MIpre | 8307 | 204.2 | 28.98 | 3931 | 1.011 | 1 |
| sub-xp107 | motorloc | 8400 | 174.2 | 28.6 | 3974 | 94.45 | 1 |
| sub-xp108 | eegfmriNF | 9030 | 166.9 | 30.69 | 7517 | 1.081 | 1 |
| sub-xp108 | eegNF | 8838 | 163.9 | 31.27 | 3787 | 1.221 | 1 |
| sub-xp108 | fmriNF | 9044 | 159.7 | 31.57 | 4424 | 1.08 | 1 |
| sub-xp108 | MIpost | 8989 | 142.6 | 32.33 | 6352 | 1.015 | 1 |
| sub-xp108 | MIpre | 9112 | 150.4 | 31.48 | 8205 | 0.9939 | 1 |
| sub-xp108 | motorloc | 9057 | 203.4 | 28.22 | 6773 | 95.45 | 1 |
| sub-xp109 | eegfmriNF | 8109 | 127.4 | 32.35 | 6780 | 1.055 | 1 |
| sub-xp109 | eegNF | 8099 | 127.3 | 33.49 | 4923 | 1.075 | 1 |
| sub-xp109 | fmriNF | 8093 | 133.9 | 32.85 | 7935 | 1.169 | 1 |
| sub-xp109 | motorloc | 7406 | 177.7 | 25.41 | 4179 | 43.2 | 1 |
| sub-xp110 | eegfmriNF | 8414 | 173.9 | 31.79 | 7284 | 1.405 | 1 |
| sub-xp110 | eegNF | 8380 | 190.8 | 31.7 | 3691 | 1.723 | 1 |
| sub-xp110 | fmriNF | 8430 | 232.2 | 32.38 | 3901 | 2.704 | 1 |
| sub-xp110 | MIpost | 8390 | 151.7 | 33 | 3072 | 1.601 | 1 |
| sub-xp110 | MIpre | 8370 | 185 | 30.97 | 3890 | 1.583 | 1 |
| sub-xp110 | motorloc | 8267 | 171.3 | 28.29 | 4642 | 51.31 | 1 |


**106/106 raw in-scanner recordings exceed the 5 mV limit** and are barred from training by the pre-registered rule. Units after correction are confirmed by the `corrected_robust_amplitude_uv` column rather than assumed from array magnitude.


## 7. Global-signal-regression ablation

GSR is reported as an explicit arm, never applied silently, because it can remove genuine broad motor responses.

| dataset | gsr_variance_removed_frac_median | gsr_variance_removed_frac_min | gsr_variance_removed_frac_max | gsr_evoked_amplitude_retained_frac_median | gsr_evoked_amplitude_retained_frac_min | gsr_evoked_amplitude_retained_frac_max |
|---|---|---|---|---|---|---|
| HEFMI | 0.1814 | 0.038 | 0.7399 |  |  |  |
| TUBerlin | 0.5247 | 0.1988 | 0.9826 | 0.1717 | 0.01706 | 216.9 |


## 8. Artifacts saved

- `qc_outputs/manifest.csv` — every recording on disk
- `qc_outputs/tables/qc_{eeg,fnirs,fmri}.csv` — one row per recording, all metrics
- `qc_outputs/tables/qc_raw_vs_corrected.csv` — in-scanner correction comparison
- `qc_outputs/tables/subject_decisions.csv` — pass/fail + reason per subject
- `qc_outputs/arrays/eeg_psd/` — PSD per recording (mean + per-channel)
- `qc_outputs/arrays/fnirs_evoked/` — event-locked HbO/HbR mean + CI
- `qc_outputs/arrays/fmri_roi/` — ROI time series + design matrix
- `qc_outputs/arrays/fmri_roi_masks/` — the ROI mask used for each subject
- `qc_outputs/figures/<dataset>/<modality>/` — per-recording diagnostic figures
