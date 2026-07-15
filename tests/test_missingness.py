from agnn.baselines.data_processing import load_participants, filter_to_modelling_cohort, add_derived_features, build_feature_matrix

df = load_participants()
df = filter_to_modelling_cohort(df)
df = add_derived_features(df)
X = build_feature_matrix(df, "compact-light")

print(f"Before dropping missing: {len(X)}")
print(f"\nMissing values per column:")
print(X.isna().sum())
print(f"\nRows with any missing: {X.isna().any(axis=1).sum()}")