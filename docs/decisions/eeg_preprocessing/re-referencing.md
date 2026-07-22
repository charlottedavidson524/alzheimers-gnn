# Re-referencing

Decided to use Common Average Reference (CAR). Apply this after bad-channel interpolation but before ICA fitting.Also considered REST (Yao 2001), linked mastoids (because it's clinical convention) and original online reference (FCz).

Three reasons for CAR:

- The 127-channel montage meets CAR's assumptions. CAR assumes that averaging across all electrodes approximates zero. This is true for dense montages covering the whole head. Only doesnt work for sparse arrays (less 32 channels or asymmetric coverage). PEARL-Neuro's 127 usable electrodes covers the whole head uniformly.

- ICLabel expects average-referenced data. Pion-Tonachini et al. (2019) trained ICLabel on data preprocessed with average referencing (EEGLAB default and convention). Using a different reference could degrade ICLabel's classification accuracy. Since automated component classification via ICLabel is core to this pipeline (see `docs/decisions/artefact-rejection.md`), matching the training data's reference is necessary

- wPLI partially insulates against reference choice. Vinck et al. (2011) developed wPLI specifically to be robust against volume-conduction esque effects, like refernce induced biases.

Why not REST:

- Zheng et al. (2018) compared REST and AR on 64-channel resting-state EEG in 38 subjects. They found no significant differences for alpha-blocking or frontal lateralization. They did report higher functional connectivity density and small-world characteristics under REST. The authors' own conclusion recommends applying both references to validate findings.

- Chella et al. (2016) compared reference schemes at four electrode densities using simulated data with imaginary coherency. they found finding REST superior to AR at low densities (21 channels) but the difference was a lot smaller at 128 channels. At 256 channels with a realistic head model, AR outperformed REST. Suggests that the 127-channel PEARL-Neuro montage sits in a regime where reference choice matters less than at low densities -> indirect support for CAR

- REST would also require building a forward head model (either from individual anatomy or a template), adds a lot of complexity. Because wPLI seems reference-robust and the empirical effect size at 127 channels is small, theres not much need for this complexity for a first pass. REST could be used for sensitivity analysis if needs be.

## Why not linked mastoids

- Qin, Yao et al. (2010) showed that linked mastoid references introduce spatial bias in connectivity analyses. In particular they inflate apparent coherence between electrodes near the mastoids.

- The group of Sheffield UK AD-EEG-GNN papers (Klepl 2022, Shan 2022, Cao 2024) used linked earlobe references and this maytches clinical convention, but newer independent AD-EEG-GNN work (Zhang & Zhu 2025) has shifted to CAR (double check). So, this decision aligns with the modern methodological consensus rather than the clinical historical convention.

## Pipeline ordering (as of now):

1. Filtering (bandpass 1-45 Hz, notch 50 Hz)
2. Bad-channel detection (LOF)
3. Bad-channel interpolation (spherical spline)
4. Common Average Reference (rereferencing)
5. Fit ICA on CAR-referenced data (using effective rank)
6. Classify components via ICLabel
7. Component removal, condition extraction, epoching

Come back to this if sensitivity analysis shows GNN results depend on reference choice -> REST. If direct comparison with Sheffield UK group results is needed, culd run linked mastoid preprocessing.
