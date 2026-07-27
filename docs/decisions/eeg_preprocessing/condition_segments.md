# Condition Segments

The choice for my condition segments was to analyse the eyes-closed segment only. This is extracted via S 4 → S 11 event markers. Eyes-open segment will be discarded from primary analysis. Also considered using eyes open only as well as concatenated eyes open and eyes closed. Both segments are analysed as separate inputs per subject.

These are the reasons for this decision:

- The PEARL-Neuro dataset authors themselves use eyes-closed. Dzianok et al. did a follow-up analysis of this dataset (Dzianok et al., 2025). They analysed APOE/PICALM effects on the exact same recordings and they used the eyes closed condition. Their paper states directly "The most common protocol used in AD patients is the 'resting-state' protocol, as it is brief and does not require participants to engage in any specific task. Most studies use eyes-closed condition, as open eyes resting-state is often characterized by an EEG desynchronization in common bands of interests." They analysed 5 minutes of eyes-closed activity per subject, epoched into 4-second segments. If I align with this convention it will make results directly comparable to the dataset authors' own work. Also good to note that Dzianok et al. found that APOE e4 carriers in this dataset showed reduced Higuchi fractal dimension (signal complexity) during eyes-closed. This is direct evidence that pre-symptomatic APOE effects are detectable in this specific dataset using this specific condition.

- APOE-EEG literature uses eyes-closed. Canuet et al. (2012), which was a reference study of APOE e4 effects on EEG in AD, recorded participants "during awake, eyes-closed state" and found APOE e4 associated with reduced alpha activity in temporo-parietal regions in AD patients. While Canuet et al. studied AD patients rather than pre-symptomatic carriers, this establishes the direction and topography of APOE e4's effect on eyes-closed alpha EEG.

- AD-EEG-GNN literature uses eyes-closed or resting-state. Klepl et al. (2023) achieved 92.9% classification accuracy (AUC 0.984) on resting-state EEG. The recent Multi-frequency GNN paper (Wang et al., 2025) uses eyes-closed. The Sheffield UK group (Shan et al., 2022; Cao et al., 2024) uses resting-state segments in a fashion consistent with eyes-closed being the standard.

- Alpha rhythm is the main AD/APOE biomarker (slowing and reduced posterior power). Alpha is suppressed when eyes are open which reduces the effect size. Eyes-closed also helps reduce eye movement and blink artefacts, is cleaner for ICA decomposition, and more stationary. This is better for wPLI connectivity estimation which assumes signal stationarity within epochs.

There is some precedent in the AD-EEG literature for not using eyes open. Alpha desynchronisation reduces AD/APOE signal amplitude. More eye-movement artefacts. Would deviate from the PEARL-Neuro authors' own choices. Jennings et al. (2022) argue eyes-open has independent diagnostic value particularly for distinguishing dementia subtypes (Alzheimer's disease vs Lewy body). The issue is that this is a different research question from APOE classification.

Shouldn't use a concatenation either. Eyes open and eyes closed produce fundamentally different EEG patterns because of alpha desynchronisation. Concatenating produces a signal that isnt either f these states, with connectivity measures becoming ambiguous. The transient at the eyes open -> eyes closed boundary would appear in the middle of the continuous signal. This could bias wPLI computation. No AD-EEG-GNN paper reviewed uses this approach.

Also considered using them both as separate inputs. Would double the sample per subject but would produce highly correlated inputs because its the same subject and the same session. This would require subject-level rather than graph-level cross-validation to avoid data leakage. It would add extra preprocessing without a clear benefit. It's also just not standard in the field. Jennings et al. (2022) suggest both eyes-open and eyes-closed EEG could be useful for dementia diagnosis, and it's particularly for determining dementia subtypes, but if i were to do this it would fundamentally change the project scope.

The extracted eyes closed segment is roughly 6 minutes (360 seconds) per subject. This is actually more than published AD-EEG-GNN work: Klepl et al. (2023) used their recordings similarly. Dzianok et al. (2025) used 5 minutes on this exact same dataset. Shouldn't worry about data quantity.

Should come back to this decision if the primary GNN model underperforms. Can add eyes open as an auxiliary analysis (follows Jennings et al. 2022's suggestion).

NOTE AFTER PREPROCESSING: Three subjects (sub-19, sub-30, sub-34) had truncated events files where the S 11 marker was missing despite the fact they had full length raw EEG recordings. As a fix the eyes closed end is inferred as S 4 + 360 seconds. This is to match the protocol duration.
