# Edge Thresholding

wPLI produces fully connected weighted 127x127 matrices per band per subject. Options are:

- No thresholding
- Absolute threshold
- Proportional/top-k threshold
- Data driven topology preserving method (MST/OMST)

This is linked to graph aggregation (per-epoch vs. per-subject thresholding are very different at n this small) and edge features.

Why thresholding is important:

- van Wijk, Stam & Daffertshofer (2010): Graph measures depend on node count and average degree in ways specific to (usually unknown) network topology, so comparisons across differently-sized/dense networks can be spurious. No correction method (fixed threshold, fixed degree, random-surrogate normalization) is fully unbiased. Threshold choice should follow from the research question, not be a default.

- van den Heuvel et al. (2017): Directly relevant to a case-control-style design like this. If overall connectivity strength differs systematically between the groups being compared, proportional thresholding forces the weaker-connectivity group to admit more low-reliability edges to hit the same fixed density. This inflates apparent between-group topological differences instead of removing a confound. Also validated on EEG phase-connectivity data (PLI), not just fMRI.

- Garrison et al. (2015): Absolute thresholds are unstable enough that the direction of a significant group difference can reverse across thresholds. Proportional thresholds are comparatively more stable, but not fully.

wPLI-specific EEG threshold sensitivity:

Adamovich, Zakharov, Tabueva & Malykh (2022): The most relevant source found. Tests threshold sensitivity on resting-state EEG across five connectivity measures including wPLI, sensor and source space, 146 recordings, proportional thresholds swept 0.01–0.99 quantile plus an OMST comparison. Key findings:

- Global graph metrics vary substantially and non-linearly with threshold (R² 0.1–0.97, median 0.62).

- wPLI and ciPLV cluster together with distinct threshold-sensitivity behavior, separate from coherence/imaginary-coherence/PPC. Thresholds calibrated on other measures don't necessarily transfer to wPLI.

- Higher thresholds reliably produce disconnected nodes. This is a real risk for GNN message-passing on a 127-node graph.

- A simulated group-difference test found the presence, absence, and even direction of a significant effect changed with threshold, including between adjacent thresholds.

- OMST vs. proportionally-thresholded graphs of matched density produced substantial, inconsistently-directed differences. OMST isn't necessarily a smarter route to the same graph.

Given the target is itself a group comparison, and Dzianok et al. already found directionally consistent but non-FDR-significant connectivity effects in this cohort, this paper's demonstration that threshold choice alone can flip significance or direction is a serious risk to manage.

Options weighed up:

The three options, evaluated

- Absolute threshold isn't a good idea. Fixed cutoff produces different effective density per subject depending on their overall connectivity strength (van Wijk et al. and confirmed for wPLI by Adamovich et al.). Risky here since APOE-related "EEG slowing" is the kind of effect that could shift overall connectivity strength between groups. this would confound density with group membership.

- Proportional (top-k%) threshold. The general-literature default (Garrison et al.) and what Klepl et al. (2022) used as one of two filters in their EEG-GNN AD-classification pipeline (k∈{10,20,30}). But van den Heuvel et al.'s caution applies directly to a case-control design: if overall wPLI strength differs by APOE group, proportional thresholding could inflate apparent group differences. This is the opposite of the property it's usually chosen for. Adamovich et al. also show it isn't internally stable for wPLI specifically.

- Data-driven, topology-preserving (MST/OMST). Guarantees full connectivity by construction (avoids the disconnected-node problem entirely) and avoids an arbitrary cutoff. Klepl et al. tested both top-k% and MST-k(1,2,3) and found their best GNN used top-20%, not MST so MST isn't automatically better for downstream classification even if it's more principled. Adamovich et al.'s OMST-vs-thresholded comparison at matched density shows OMST is a genuine alternative, not a refinement of proportional thresholding. Dimitriadis et al. (2017) which is OMST's originating paper (tested on iPLV, not wPLI, on a different task) also found OMST massively outperformed absolute/proportional/mean-degree thresholding on EEG subject-identification accuracy (99% vs. 61–82%) and fMRI test-retest reliability (ICC 0.89–0.91 vs. 0.53–0.61). These are different use cases than what im doing but its independent evidence OMST is a well-validated method worth including for comparison

Conclusion: use proportional (top-k%) thresholding as the default with k as a hyperparameter. Gate it on an empirical check of whether overall wPLI strength differs by APOE group.

- A fully-connected graph isn't obviously safer (Klepl et al. found edge-filtered graphs outperformed unfiltered input for their best GNN).

- Nothing in the lit favours absolute thresholding

- Proportional thresholding's main failure mode (van den Heuvel et al.) is conditional and checkable: compute mean/median subject-level wPLI per band and compare by APOE group before finalizing. If overall strength is comparable across groups, proportional thresholding is reasonable and consistent with Klepl et al.'s precedent. If it differs significantly, document this as a limitation and weight MST-based filtering more heavily instead.

- Given Adamovich et al.'s finding that threshold choice can flip effect direction/significance, k should be an ablation axis (10/20/30%, matching Klepl et al. for comparability) rather than a single fixed guess

- Run MST/OMST as a genuine comparator, not an afterthought. Klepl et al. and Adamovich/Dimitriadis et al. point in different directions on whether it helps, so this is an open question best settled using own data.

- Decide a disconnected-node handling rule upfront regardless of method chosen. Drop isolated nodes (likely incompatible with a fixed 127-node topology across subjects as per frequency-bands.md's edge-vector proposal), keep them with a floor value, or use MST/OMST to guarantee connectivity by construction. Decide after fleshing out node features and graph aggregation plans.

GIVEN TIME SCONSTRAINTS, CHOOSE: proportional threshold at k=20% (do a hyperparameter sweep for k and MST also after the main model). This dge-threshold sensitivity ablation is the cheapest to run (same graphs, same architecture, just re-threshold) and it's the one Adamovich et al. showed can flip results
