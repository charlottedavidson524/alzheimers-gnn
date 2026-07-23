Needed to decide what electrode positions to use for theis analysis. As discussed in `docs/glossary/eeg-electrode-layout.md`, either MNE's `standard_1005` layout could be used, or each participan'ts individual CapTrak data.

Decided to use `standard_1005` over CapTrak.

The GNN pipeline is in sensor space. Treats each of the 127 electrodes as a graph node and computes functional connectivity edges between electrode pairs. This is not the same as from source-space analysis like LORETA or beamforming, which require precise coregistration of electrode positions to individual brain anatomy for inverse modelling to be accurate.

The lierature on electrode position is mostly focused on source localization (Homölle and Oostenveld (2019), Dalal et al., 2014). While they identify the importance of precise electrode positioning in this specific circumstance, I am not focusing on this circumstance. These findings are for source-space analyses not sensor-space connectivity. Since this project focuses on sensor-space functional connectivity, and the literatuire in this area uses template positions (Klepl et al., 2022, Shan et al., 2022), I am justified in selecting this.

Should revisit if pipeline gets extended to include source-space analyses (eg source-localised connectivity as a sensitivity check, or region of interest-based node definitions)

NOTE AFTER PREPROCESSING SUB-01: O9 and O10 dont have 3D positions in MNE's standard_1005 template. Dropped from every subject at the start of preprocessing. This gives a 125-node graph for the cohort
