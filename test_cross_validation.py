"""
Quick file to test that cross validation code works.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from agnn.evaluation.cross_validate import cross_validate, format_summary

# Generate synthetic dataset. Easily learnable on purpose.
rng = np.random.default_rng(42)
X = rng.standard_normal((100, 5))
y = (X[:, 0] > 0).astype(int)

# Test the cross_validate() function
result = cross_validate(lambda: LogisticRegression(), X, y)

# Print the outputs
print("Per-fold:")
print(result["per_fold"])
print("\nSummary:")
print(format_summary(result["summary"]))