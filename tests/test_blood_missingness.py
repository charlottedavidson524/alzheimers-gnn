from agnn.baselines.data_processing import load_participants, filter_to_modelling_cohort

df = load_participants()
df = filter_to_modelling_cohort(df)

print(f"N with second_phase=1: {len(df)}")
print(f"N with leukocytes present: {df['leukocytes'].notna().sum()}")
print(f"N missing leukocytes: {df['leukocytes'].isna().sum()}")
print(f"N with total_cholesterol present: {df['total_cholesterol'].notna().sum()}")