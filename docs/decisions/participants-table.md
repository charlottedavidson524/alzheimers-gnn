Some notes from the first inspection of participants.tsv.

Confirmed data structure:

- 192 participants x 87 columns, as in the paper (Dzianok & Kublik, 2024).
- All expected variables were present: genertics (APOE_rs429358, APOE_rs7412, APOE_haplotype,
  PICALM_rs3851179), demographics (age, sex, education, BMI), health/lifestyle (smoking, coffee, allergies, hypertension, diabetes, thyroid disease, ibuprofen intake, dementia history, learning deficits), psychometrics (BDI (depression), SES (stress), RPM (intelligence), EHI (handedness), NEO (personality factors), AUDIT (alcohol use), MINI-COPE_1..14 (coping), CVLT_1..13 (verbal learning/memory)), blood (full CBC (leukocytes, erythrocytes, hemoglobin, hematocrit, MCV, MCH, MCHC, RDW-CV, platelets, PDW, MPV, P-LCR, differential counts and percentages); lipid panel (total/HDL/non-HDL/LDL cholesterol, triglycerides); HSV_r (herpes simplex antibody))

Column names used:

- participants.tsv uses specific naming conventions that could differ from BIDS defaults. Names are as follows:

| Concept                            | Column Name        |
| ---------------------------------- | ------------------ |
| APOE genotype (combined haplotype) | `APOE_haplotype`   |
| PICALM rs3851179 genotype          | `PICALM_rs3851179` |
| Age                                | `age`              |
| Sex                                | `sex`              |

The two individual APOE SNPs (`APOE_rs429358`, `APOE_rs7412`) are present but not directly used because `APOE_haplotype` is the combined e2/e3/e4 genotype that determines carrier status. It's the cleaner column.

No EEG/fMRI flag column:

- participants.tsv doesn;t have a column showing which participants have neuroimaging data. The neuroimaging
  subset needs to be identified by checking which sub-XX/ folders have EEG/fMRI files on disk. The script handles this by reporting the absence and skipping the subject specific analysis.
