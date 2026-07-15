from agnn.baselines.data_processing import load_participants, filter_to_modelling_cohort, add_derived_features, build_feature_matrix, impute_missing, drop_missing


df = load_participants()
df = filter_to_modelling_cohort(df)
df = add_derived_features(df)
X = build_feature_matrix(df, "all-features-light")
X = impute_missing(X)
X, _ = drop_missing(X, df["APOE_haplotype"])

print("Column dtypes:")
print(X.dtypes.to_string())
print("\nNon-numeric columns:")
for col in X.columns:
    if X[col].dtype == "object":
        print(f"{col}: sample values = {X[col].unique()[:5].tolist()}")