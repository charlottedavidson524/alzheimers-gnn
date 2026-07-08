"""
Extended exploratory analysis of participants.tsv.

Goals of this file:

- Cross tab second_phase against blood-panel availability to test sub-cohort hypothesis.
- Rank numeric features by their association with APOE e4 carrier status (effect size and FDR-corrected p-values)
- Plot correlation heatmaps of the blood panel and psychometric feature blocks.
- Cross-tab potential confounders (education, smoking, BMI category, AUDIT category, family history) against carrier status

Outputs will be saved in results/extended_exploration/

USAGE
-----
From the project root:
    - python scripts/extended_exploration.py
"""

import sys
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
 
from agnn.config import load_config