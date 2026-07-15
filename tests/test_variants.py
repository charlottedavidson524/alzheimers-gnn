from agnn.baselines.data_processing import load_baseline_data

for variant in ["compact-light", "compact-full", "all-features-light", "all-features-full"]:
    X, y, feature_names = load_baseline_data(variant)
    n_pos = int(y.sum())
    n_neg = int((1-y).sum())
    print(f"{variant:22s}: shape={X.shape}, "f"carriers={n_pos}, non-carriers={n_neg}, "f"pos_rate={100*y.mean():.1f}%")