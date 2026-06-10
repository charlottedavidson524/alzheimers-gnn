These findings come from running scripts/explore_participants.py blocks 4-6 on the full 192 participant sample.

APOE e4 carrier balance (from the full sample):

- e4 carriers: 51 (26.6%)
- e4 non-carriers 141 (73.4%)

This is moderately imbalanced but is also in line with what can be expected of the e4 prevalence in the Polish population. The Polish gen-pop APOE e4 allele frequency is around 10.6% (Bednarska-Makaruk et al, 2001). This predicts a carrier rate of roughly 20% under Hardy-Weinberg. PEARL-Neuro's sample observed carrier rate of 26.6% is therefore higher than the Polish baseline. This is also consistent with the study's goal of recrutiing participants who have higher dementia risk. This higher proportion supports the study's framing as a pre-symptomatic risk-carrier cohort but it should be acknowledged ragarding generalisability.

Stratified k-fold cross validation should be used to preserve class proportions across folds. F1, balanced accuracy and ROC_AUC are potential metrics. Raw accuracy can't be used with this level of imbalance. Non-carrier would always be predicted with 73% accuracy.

APOE haplotype distribution across the whole sample:

| Genotype | Count |
| -------- | ----- |
| e3/e3    | 119   |
| e3/e4    | 46    |
| e3/e2    | 21    |
| e2/e4    | 3     |
| e4/e4    | 2     |
| e2/e2    | 1     |

Only two homozygous e4/e4 participants. They have the highest genetic Alzheimer's risk.

PICALM rs3851179 distribution from the full sample:

| Genotype | Count |
| -------- | ----- |
| G/G      | 79    |
| G/A      | 97    |
| A/A      | 16    |

G is the Alzheimer's risk allelle. G/G is the highest-PICALM-risk group. Broadly Hardy-Weinberg possible.

APOE and PICALM joint distribution:

The two genetic risk factors are independent. Cross-tab shows no strong association between APOE e4 carrier sttaus and PICALM genotype. So, the two markers carry potentially complmentary risk info. Should frame both as inputs to the analysis.

Whitespace in genotype strings:

There were some APOE and PICALM entries which had trailing whitespace. This amde identical genotypes count as two separate entries. Before treating any genotype column as categorical, must .str.strip() it.
