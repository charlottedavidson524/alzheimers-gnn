# Functional Connectivity Findings -> Full Cohort

## Overall Performance

- 77 subjects succeeded the pipeline with no errors.
- All had two bands present (delta and alpha-2)
- All 125 nodes
- All 3100 edges per band because top 20% were symmetrised

## Runtime Distribution

- The median subject took around 55 seconds
- The range of time taken was 34s (sub-76) to 58s (sub-31)
- Variation in time taken was driven mostly by epoch count

## Data Volume

- There were 74 subjects with 90 epochs. 180 graphs each.
- There were 3 subjects with fallback duration (70 epochs). they produced 140 graphseach. sub-19, sub-30, sub-34
- There were 5 subjects with high autorejection:
  - sub-36: 128 graphs (64 epochs)
  - sub-54: 122 graphs (61 epochs)
  - sub-70: 152 graphs (76 epochs)
  - sub-74: 144 graphs (72 epochs)
  - sub-76: 112 graphs (56 epochs) -> lowest in the cohort
- The total number of graphs across the cohort is around 13,500 (77 subjects by roughly 180 graphs. this is adjusted for reduced epoch subjects)

## APOE Label Distribution

- Non-carriers (label=0): sub-01 through sub-31. This was 31 subjects (40%)
- Carriers (label=1): sub-32 through sub-79 (minus sub-55 and sub-69). This was 46 subjects (60%)

Class imbalance ratio is 1.48:1 of carriers to non-carriers. Mild class imbalance, can be managed.

## Notable Subjects

Two subjects have unusually reduced data:

- sub-76: 56 epochs and 112 graphs. Above the adequacy threshold but it's the smallest sample in the cohort
- sub-54: 61 epochs and 122 graphs. Similar situation as above

Both had high autoreject rejection during preprocessing (can see this in the preprocessing notes). They still produce valid graphs but the GNN will just see fewer training examples from them.
