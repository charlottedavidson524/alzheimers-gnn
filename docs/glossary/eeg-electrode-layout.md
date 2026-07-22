# EEG Electrode Layout

This is a reference for the 128 electrode EEG montage used in PEARL-Neuro and it's implications for GNN graph construction. Based on Figure 3 of the paper, and it's methods section, as well as some insights from inspecting the first participant's .vhdr file.

## Equiptment

PEARL-Neuro recorded EEG using an actiCHamp electrode cap (Brain Products GmbH, Munich, Germany). These are standard Brain Products research grade EEG systems. They used a factory-specified layout, not a custom arrangement. Recorded settings from the paper were:

- Sample rate: 1000Hz
- Online reference: FCz
- Online filters: low pass ar 280Hz only. No notch, no high-pass filter applied during recording. Filtering choices fall under preprocessing.
- Impedence: 5-10 ohms on average. Maintained by skin abrasion and gel application.

## Naming system: extended 10-5

Electrode names in Figure 3 (e.g. Fp1, Cz, FFC5h, CCP3h and PPO9h) follow the extended 10-5 international system (Oostenveld and Praamstra, 2001). This is the densest standard naming scheme. It builds on older systems.

- 10-20 (Jasper, 1958) is a 21 electrode scheme
- 10-10 system (adds intermediate posistions for higher density, roughly 70 electrodes)
- 10-5 system = adds half-distance electrodes between 10-10 sites for very high density (up to 345 positions). These are the names ending in h, like FFC5h. h literally means half. FFC5h is halfway between FFC5 and FFC3 in 10-10 grid.

## Reading electrode names

Each name encodes a brain region (letter prefix) and a position along the left-right axis (number or z). Region prefixes are:

- Fp = frontal pole (above eyebrows, frontmost)
- AF = anterior frontal
- F = frontal
- FC/FT = fronto-central/fronto-temporal
- C = central (over the central sulcus/motor strip)
- T = temporal (over temporal lovbes, lateral)
- CP/TP = centro-parietal/temporo-parietal
- P = parietal
- PO = parieto-occipital
- O = occipital (back of head and over the visual cortex)

Double letter prefixes like FFC or PPO mean 'between F and FC' or ' between P AND PO'

The number conventions are:

- Odd numbers = left hemisphere
- Even numbers = right hemisphere
- z suffix (Fz, Cz, Pz, Oz, etc) = midline
- Larger numbers = further from midline. F3 is closer to centre than F7 for example.

Take FC4: fronto central, right-hemisphere, moderately lateral

## Two rings in Figure 3

Inner ring and central grid are the standard 10-10/10-5 layout covering the top and sides of head. The outer ring (e.g. F9, F10, FT10) are below the standard ring. They pick up signal from lower scalppositions like ears, mastoids, etc.

## 127 electrdoes

- Paper describes 128 electrode cap but data files have 127 channels. Missing FCz.
- PEARL_Neuro paper explains this. FCz was online refercne electrode during recording. Confirmed this empirically in sub-01's data (see `docs/findings/eeg-inspection-sub01.md`)
- Fz and Cz could be called by the code. |These are FCz's immediate neighbours. Only FCz is missing so it is the reference.
- Implications for the project: brain graphs will ahve 127 nodes not 128. `standard_1005` MNE montage won't apply a position to FCz (no problem as electrode isn't in data anyway). Signal at every recorded electrode is relative to FCz at the moment. re-referencing will change this.

## Electrode positions

PEARL-Neuro provides per-subjec electrode positions. CapTrak per subject co-ordinates stored in `sourcedata/sub-XX/coords/*.sfp`on OpenNeuro.

- Per subject measured electrode positions. More accurate. reflect participants cap fit and head shape.
- Missing sub-30 and sub-51.

MNE's `standard_1005` template.

- Verified in the sub-01 inspection and matches all 127 recorded channels clanly.
- Simpler than CapTrak but is idealised positions, not measured ones. Samll cuuracy loss.

| Property                  | CapTrak                     | `standard_1005`                |
| ------------------------- | --------------------------- | ------------------------------ |
| Accuracy                  | High                        | Moderate                       |
| Setup complexity          | Higher                      | Trivial                        |
| Coverage                  | 77/79                       | 79/79                          |
| Extra download?           | Yes                         | No                             |
| Cross-subject consistency | Real anatomy variation kept | All subjects use same position |

`standard_1105` is probably fine for a first pass.

- Switch to CapTrak if downstream analyses are sensitive to electrode positions (e.g. source-localised connectivity).

Implications for graph construction:

- 127 electrodes become 127 nodes of each participants's brain graph.
- Node identity is fixed across participants. Good for GNN training -> model can learn certain nodes tejd tyo carry certain signal across participants. Sanity check when running EDA.
- Spatial proximity is relevant due to volume conduction. Raw coherence between electrodes very close can be spurious. Connectrivity measures need to be robust to volume conduction (e.g. wPLI. imaginary coherence). Diatnce based edge thresholding is a possible mitigation. Can drop edges between electrodes within certain distances of each other. Avoids volume conduction.

## 3D co-ords as node features

For GNNs, each node can carry a 3D co-ordinate as a feature. Lers model learn spatial-relationships. Co-ordinates are in metres for a head-centred co-ordinate system.

Citation: Oosterveld, R. and Praamstra, P. (2001) 'The five percent electrode system for high-resolution EEG and ERP measurements', Clinical Neuropsychology, 1 12(4), pp.713-719.

^ The original 10-5 system paper
