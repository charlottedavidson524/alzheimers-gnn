# Graph Aggregation

Needed to choose whether the GNN sees one summary graph per subject or many per-epoch graphs. Important to also consider how that interacts with train/test splitting

After preprocessing, each subject has around 90 clean 4-second epochs of eyes-closed resting EEG. wPLI can be computed either once per subject on an aggregated/averaged signal which would give one summary graph per subject, or separately per epoch, giving roughly 90 graphs per subject. Second option builds on n=77 subjects into N≈77×90 training graphs. This is a big jump in sample size but it does result in strong statistical dependence between graphs from the same subject. Need to bear in mind that this has consequences for how train/test splitting needs to be done.

Decided to do per-epoch graphs as separate training samples. Per-epoch graphs are used as the default, matching the closest thing to a precedent in the GNN-EEG-AD-classification literature (Klepl et al., 2022, 2023). They split EEG into 3-second segments (Klepl et al., 2022) or 1-second overlapping windows (Klepl et al., 2023) and trained on each one as an independent graph. This makes GNN training possible at a small sample size (their N=40 total vs my N=77 subjects). This choice has an important knock on effect, see below.

SUBJECT LEVEL SPLITTING ONLY. If per-epoch graphs are used, no subject's epochs can EVER be split across training and test/validation sets. This is vital and the literature is very clear. This point is made most clearly on an Alzheimer's EEG classifier.

Brookshire et al., (2024). They directly compared segment-based holdout (epochs from one subject can appear in both train and test) against subject-based holdout (all of one subject's epochs held to a single partition). They used a CNN trained to distinguish Alzheimer's disease from subjective cognitive impairment on resting EEG (roughly the same type of task as this project). Result -> segment-based holdout gave 99.8% test accuracy but subject-based holdout (identical model and data) had 53.0% accuracy. Not statistcially any different from chance. A second experiment (epileptic seizure detection, a within-subject task) showed the same direction of bias. This one was smaller in magnitude (79.1% segment-based vs. 65.1% subject-based). They believe larger effect in the Alzheimer's (between-subject) task is becayse of the model being able to basiclly learn "which subject is this" as a shortcut to the diagnosis label. This isnt a shortcut that would be available in a within-subject task. Their review of 63 published translational DNN-EEG studies found only 27% unambiguously used correct subject-based holdout.

Can apply this directly. Epochs from the same subject resemble each other more than epochs from different subjects, because individual patterns dominate over the disease/risk-relevant signal. This applies equally to a GNN classifying APOE e4 status from per-epoch connectivity graphs. Could argue that it's even more of a risk for a genetic-risk target than a diagnosis target, since APOE status is an even more fixed, purely individual-level label. Klepl et al. (2022, 2023) use stratified group k-fold cross-validation with subject ID as the group. Keeps every one of a subject's segments in a single fold.

There is a solid implementation requirement/. need to generate a single subject-ID-to-fold mapping once, and reuse it everywhere epochs are split, ie training/validation/test partitioning, cross-validation folds, and any hyperparameter selection. (Hyperparameter tuning against a validation set drawn from the same subjects used in training is a leakage risk if that same validation performance is then reported as the final result). A nested or subject-disjoint validation split is needed if hyperparameters are tuned at all.

For some useful context on why more epochs isn't always just better and why per-epoch graphs being individually noisier than an aggregated graph is not a surprise:

If time remains after the primary pipeline and fusion-architecture comparison can run a per-subject-summary-graph comparison experiment. Could be very informative. Tests if the per-epoch default actually outperforms the simpler, more directly matched to target label alternative (one graph per subject beecasue classification target is per-subject).

Things that need to be gotten right for implementation:

- Every epoch from a subject carries that subject's ID as metadata through preprocessing, graph construction, and dataset assembly.

- A single train/val/test (or CV fold) assignment is made at the subject level before any epoch-level graphs are generated. That assignment is the only place subject-to-partition logic lives. It's not re-derived independently in multiple scripts.

- At inference/evaluation time, per-epoch predictions for a held-out subject need to be aggregated into a single subject-level prediction (because the classification target is per-subject, not per-epoch).

- A simple approach (majority vote or averaged predicted probability across that subject's epochs) is good enough as a default

- If hyperparameters are tuned at all, use a validation split that is subject-disjoint from both the training and final test sets, and do not report validation-set performance as the final result.
