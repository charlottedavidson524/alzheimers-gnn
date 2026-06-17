These outputs come from running `scripts/explore_participants.py` blocks 7-9 on the full n=192 sample. Builds on `docs/findings/genetic-profile.md` which ran blocks 4-6.

As documented in `docs/decisions/participants-table.md`, `participants.tsv` doesn't contain an EEG/fMRI flag column. The n=79 subset needs to be identified from on-disk subject folders after stage 3 of the download. This is confirmed bt the block 7 output.

Missing data:

- 57/87 columns have missing values.
- Biggest missing data cluster has 116 out of 1192 participants missing.

18 of the top 20 most missing columns have 116 missing values. These are all blood test variables, indictaing structure to the missing data (these participants did not take a blood test). The two other missingness groups are allergies (119) and session_order (122)

Age demographics:

- Mean: 55.05 years
- SD: 3.08 years
- Range: 50–63
- IQR: 52–57

Matches the paper exactly. Good in that it reduces age as a confounding variable but it also limits generalisability.

Gender demographics:

- 0: 89
- 1: 103

Roughly a 46/54 split, not too unbalanced. Need to check which way around this is although I suspect 1 is female as they are more likely to volunteer (find a source for this)
