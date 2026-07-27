# Connectivity Measure

Decided to use weighted Phase Lag Index (wPLI) as the main functional connectivity measure for graph construction.It was also important to look at spectral coherence, Phase Locking Value (PLV), Phase Lag Index (PLI), the imaginary part of coherency (ImC) and amplitude envelope correlation (AEC).

there were four reasons for wPLI:

- Robustness to volume conduction. Sensor-space EEG is affected by volume conduction. The same neural source contributes to multiple electrodes -> produces zero-lag correlations that look like connectivity. wPLI operates on the phase-lag distribution between two signals. Since volume conduction from a single source produces zero-lag coupling, phase lag based measures like wPLI isolate genuine non instant communication between regions. Vinck et al. (2011) build on this principle (established by Nolte et al. 2004 for imaginary coherency, extended by Stam et al. 2007 for PLI) by weighting phase leads and lags by the magnitude of the imaginary component of the cross-spectrum. This reduces sensitivity to volume conduction and noise. The preprocessing pipeline uses CAR referencing (see `docs/decisions/re-referencing.md`), which reduces but does not eliminate volume conduction. wPLI provides another way of dealing with this at the graph-construction stage.

- Robust to noise and more statistical power than PLI. Vinck et al. (2011) say : "two advantages of the WPLI over the PLI, in terms of reduced sensitivity to additional, uncorrelated noise sources and increased statistical power to detect changes in phase-synchronization." wPLI weights phase leads/lags by the imaginary cross-spectrum magnitude. This addresses PLI's discontinuity problem. Small phase disturbances can flip lags to leads in PLI, which wPLI's weighting scheme dampens. For a small cohort (N=79) analysing subtle pre-symptomatic APOE effects, statistical power matters directly.

- Robust against referencing, which helps with the preprocessing I have set up. wPLI is one of the most reference robust phase measures (alongside PLI and imaginary coherence). So, connectivity estimates arent very affected by the choice of re-referencing method. This fits with the CAR referencing decision (see `docs/decisions/re-referencing.md`), where wPLI's reference-robustness is names as a reason for allowing CAR's slight disadvantage compared to REST.

- Literature convention. wPLI is used across the AD-EEG-GNN literature: Klepl et al. (2023, AGGCN paper) uses wPLI as one of the primary measures, Klepl et al. (2022) includes wPLI in their systematic comparison of 8 methods, and recent multi-frequency GNN work uses wPLI. Klepl et al. (2022) compared 8 connectivity measures for AD-EEG-GNN classification and found that "no FC measure performs consistently better than the other measures." So, the choice cant be defended using a benchmark, it needs to be defended using theory (volume conduction, noise robustness, statistical power), see above

Why not coherenece:

- Spectral coherence is very sensitive to volume conduction. This produces spurious connectivity between electrodes that are close spatially. Klepl et al. (2022) included coherence in their comparison but it did not perform better than wPLI.

Why not PLV:

Phase Locking Value is amplitude-independent but its still sensitive to volume conduction. wPLI addresses this specific issue that PLV cannot. Could add PLV as a secondary computation but its not the main measure.

Why not PLI:

The theoretical predecessor to wPLI. Vinck et al. (2011) introduced wPLI as an improvement on PLI, addressing PLI's discontinuity problem and reduced statistical power. Vinck et al. also show that PLI's direct estimator is positively biased at small sample sizes. Using PLI when wPLI is available would be nonsensical.

Why not imaginary coherence:

Imaginary coherence is a legit alternative. Its also volume conduction robust and reference robust. wPLI is preferred because its more established in the AD-EEG-GNN literature, Vinck et al. (2011) note that PLI (and by extension wPLI) "improves on the ImC" by being less influenced by phase delays, and Klepl et al. (2022) compared both and neither dominated, but wPLI has more downstream tooling in MNE-Python

Why not AEC:

Amplitude Envelope Correlation captures amplitude-based coupling rather than phase-based. This is a different kind of coupling, which is a valid research question but its not the primary target for this project. The AD-EEG literature usually focuses on phase-based measures because alpha slowing ( main AD biomarker) is a phase-related thing.

If AD/APOE classification performance is bad, consider:

- Adding a supplementary imaginary coherence computation as a robustness check
- Investigating whether specific frequency bands would benefit from different measures
- Testing amplitude envelope correlation as a complementary measure (co-activation vs phase-synchronization)
- Switching to debiased wPLI (dwPLI) if sample-size bias appears to affect low-epoch subjects
