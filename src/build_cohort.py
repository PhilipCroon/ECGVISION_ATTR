# %%
"""
Cohort construction + train/test split for the ECGVISION-ATTR retrain.

Ported from multimodal_amyloid/code/cohort_yale.py. Same logic, two changes:
  1. PyP source is the new parquet (project.pyp_file, March-2026 update).
  2. cutoff_date = 2025-07-01 (later -> more training data, small temporal test).

Groups produced:
  amyloid      = PyP "consistent with TTR cardiac amyloidosis" (+ tafamidis MRNs)
  pyp_negative = PyP "not consistent"
  control      = echo LVH (Composite_LVH_binary==1) or severe AS (SevereAS_ObjSubj==1),
                 excluding any PyP-case MRN

Outputs: tabs/cohort_train.csv, tabs/cohort_test.csv  (ECG level, one FileID per row)
The R matching step (matching.R) consumes cohort_train.csv.
"""

# %% === Step 0: Prep ===
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project

# Demographics for MRN -> BIRTH_DATE (same sources as research repo)
demo = pd.read_csv(
    os.path.join(project.s3_implementation, 'CarDS_2435227_Patients.txt'),
    sep='\t', on_bad_lines='skip', low_memory=False)

path_amy_patients = ('/home/pmc57/s3_implementation/cardsjdat-CC1022-MEDINT/'
                     '2356781-CarDS-Aim-1/2356781-CarDS-Aim-1-Amyloid/'
                     'Data-2025-08-20/CarDS_2356781_Amyloid_Patients.txt')
demo_amyloid = pd.read_csv(
    path_amy_patients, sep='\t', on_bad_lines='skip', low_memory=False)

demo = pd.concat([demo, demo_amyloid])
demo['MRN'] = demo['PAT_MRN_ID']

cutoff_date = pd.to_datetime(project.CUTOFF_DATE)


def clean_sex(x):
    x = str(x).strip().lower()
    if x in ['m', 'male']:
        return 'M'
    elif x in ['f', 'female']:
        return 'F'
    return 'U'


# %% === Step 1: Load and clean ===
# NEW PyP data update (parquet, not csv)
pyp = pd.read_parquet(project.pyp_file)

if 'MRN' not in pyp.columns and 'PAT_MRN_ID' in pyp.columns:
    pyp = pyp.rename(columns={'PAT_MRN_ID': 'MRN'})

# Age_pyp: the new parquet may not carry an 'Age' column (Image_alejandra.py
# never references it). Guard so a missing column is survivable -- Age has
# downstream fallbacks (ECG-derived, then PatientAge_ECGData).
pyp['Age_pyp'] = pyp['Age'] if 'Age' in pyp.columns else np.nan

# --- Add tafamidis-treated patients as PyP-positive cases ---
taf = pd.read_csv(project.tafamidis_file, dtype=str)
if 'MRN' not in taf.columns and 'PAT_MRN_ID' in taf.columns:
    taf = taf.rename(columns={'PAT_MRN_ID': 'MRN'})
taf['first_med_date'] = pd.to_datetime(taf.get('first_med_date'), errors='coerce')

pyp_mrns = set(pyp['MRN'].dropna().astype(str).unique())
taf_mrns = set(taf['MRN'].dropna().astype(str).unique())
to_add = sorted(list(taf_mrns - pyp_mrns))
print(f"TAF -> MRNs to add at pyp level: {len(to_add)}")

if to_add:
    taf_date_map = taf.set_index('MRN')['first_med_date'].to_dict()
    new_rows = []
    for mrn in to_add:
        row = {c: pd.NA for c in pyp.columns}
        row['MRN'] = mrn
        if 'Study Date' in pyp.columns:
            dt = taf_date_map.get(mrn, pd.NaT)
            row['Study Date'] = (pd.Timestamp(dt).strftime('%Y-%m-%d')
                                 if not pd.isna(dt) else pd.NA)
        row['YH_PYPInterpretation'] = 'consistent with TTR cardiac amyloidosis'
        new_rows.append(row)
    new_df = pd.DataFrame(new_rows, columns=pyp.columns)
    pyp = pd.concat([pyp, new_df], ignore_index=True, sort=False)

print(f"pyp now has {pyp.shape[0]:,} rows and {pyp['MRN'].nunique():,} unique "
      f"MRNs (added {len(to_add)}).")

# ECG metadata + echo outcomes reused from research repo (PyP-only update)
ecgs = pd.read_csv(project.ecg_metadata_file)
echo = pd.read_csv(project.echo_outcomes_file)

# Remove patients with an amyloid ICD code (avoid contaminating controls)
echo = echo[echo['Amyloid_ICD'] != 1]

# %%
pyp['pyp_date'] = pd.to_datetime(pyp['Study Date'], errors='coerce')
pyp['PyP_conclusion'] = pyp['YH_PYPInterpretation'].map({
    'consistent with TTR cardiac amyloidosis': 1,
    'not consistent with TTR cardiac amyloidosis': 0,
})

ecgs['ECGDate'] = pd.to_datetime(ecgs['ECGDate'], errors='coerce')

# Step 2: Merge BIRTH_DATE from demo into ecgs, compute Age
ecgs = ecgs.merge(demo[['MRN', 'BIRTH_DATE']], on='MRN', how='left')
ecgs['BIRTH_DATE'] = pd.to_datetime(ecgs['BIRTH_DATE'], errors='coerce')
ecgs['ECGDate'] = pd.to_datetime(ecgs['ECGDate'], errors='coerce')
ecgs['Age'] = ((ecgs['ECGDate'] - ecgs['BIRTH_DATE']).dt.days // 365).astype("Int64")
print(f"ECGs with missing Age: {int(ecgs['Age'].isna().sum())}")

echo['EchoDate'] = pd.to_datetime(echo['EchoDate'], errors='coerce')

# Deduplicate PyP: prefer positive, then most recent.
# .fillna(0) on PyP_conclusion so unmapped interpretations don't poison priority.
pyp['sort_priority'] = (
    pyp['PyP_conclusion'].fillna(0) * 10_000
    + pyp['pyp_date'].astype('int64') // 1e9
)
pyp = pyp.sort_values(['MRN', 'sort_priority']).drop_duplicates('MRN', keep='last')

echo = echo.sort_values(by=['MRN', 'EchoDate']).drop_duplicates(subset='MRN', keep='last')

ecgs['PatientSex_ECGData'] = ecgs['PatientSex_ECGData'].apply(clean_sex)

# %% === Step 2: Merge PyP + ECG ===
pyp_ecg_all = pd.merge(
    pyp[['MRN', 'pyp_date', 'Age_pyp', 'PyP_conclusion']],
    ecgs, on='MRN')
pyp_ecg_all['Age'] = pyp_ecg_all['Age'].fillna(pyp_ecg_all['Age_pyp'])
pyp_ecg_all['PatientAge_ECGData'] = (
    pyp_ecg_all['PatientAge_ECGData'].astype(str).str.extract(r'(\d+)').astype(float))
pyp_ecg_all['Age'] = pyp_ecg_all['Age'].fillna(pyp_ecg_all['PatientAge_ECGData'])

pyp_ecg_all['days_diff'] = (pyp_ecg_all['ECGDate'] - pyp_ecg_all['pyp_date']).dt.days
print(f"Amyloid MRNs before ECG window: "
      f"{pyp_ecg_all[pyp_ecg_all['PyP_conclusion'] == 1]['MRN'].nunique()}")
pyp_ecg_all = pyp_ecg_all[pyp_ecg_all['days_diff'] >= -365]
print(f"Amyloid MRNs after ECG window (>=-365d): "
      f"{pyp_ecg_all[pyp_ecg_all['PyP_conclusion'] == 1]['MRN'].nunique()}")

# %% === Step 3: Split PyP into Train/Test by date ===
pyp_test_all = pyp_ecg_all[pyp_ecg_all['pyp_date'] > cutoff_date].copy()
pyp_train_all = pyp_ecg_all[~pyp_ecg_all['MRN'].isin(pyp_test_all['MRN'])].copy()

amyloid_train = pyp_train_all[pyp_train_all['PyP_conclusion'] == 1].copy()
pyp_negatives_train = pyp_train_all[pyp_train_all['PyP_conclusion'] == 0].copy()
amyloid_test = pyp_test_all[pyp_test_all['PyP_conclusion'] == 1].copy()
pyp_negatives_test = pyp_test_all[pyp_test_all['PyP_conclusion'] == 0].copy()

# Surface the split so the test arm is confirmed small-but-nonzero.
print(f"\n[Split @ {cutoff_date.date()}]")
print(f"  amyloid+ TRAIN MRNs: {amyloid_train['MRN'].nunique()}")
print(f"  amyloid+ TEST  MRNs: {amyloid_test['MRN'].nunique()}")
print(f"  pyp_neg  TRAIN MRNs: {pyp_negatives_train['MRN'].nunique()}")
print(f"  pyp_neg  TEST  MRNs: {pyp_negatives_test['MRN'].nunique()}")

# %% === Step 4: Echo controls (LVH / severe AS) ===
echo = echo[['MRN', 'EchoDate', 'Composite_LVH_binary', 'SevereAS_ObjSubj']]
echo_ecg = pd.merge(echo, ecgs, on='MRN')
echo_ecg = echo_ecg[~echo_ecg['MRN'].isin(pyp_ecg_all['MRN'])]
echo_ecg['days_diff'] = (echo_ecg['ECGDate'] - echo_ecg['EchoDate']).dt.days

candidate_controls = echo_ecg[echo_ecg['days_diff'].abs() <= 30].copy()
case_mrns = set(pyp_ecg_all['MRN'])
candidate_controls = candidate_controls[~candidate_controls['MRN'].isin(case_mrns)].copy()
print(f"\nCandidate control MRNs: {candidate_controls['MRN'].nunique()}  "
      f"rows: {candidate_controls.shape[0]}")

# Assign each control MRN to train/test by its latest ECG date (tie -> train)
ecg_latest = (candidate_controls.sort_values('ECGDate')
              .groupby('MRN', as_index=False).last())
control_test_mrns = set(ecg_latest[ecg_latest['ECGDate'] > cutoff_date]['MRN'])
control_train_mrns = set(ecg_latest[ecg_latest['ECGDate'] <= cutoff_date]['MRN'])

control_train = candidate_controls[candidate_controls['MRN'].isin(control_train_mrns)].copy()
control_test = candidate_controls[candidate_controls['MRN'].isin(control_test_mrns)].copy()
print(f"Control train MRNs: {len(control_train_mrns)}  rows: {control_train.shape[0]}")
print(f"Control test  MRNs: {len(control_test_mrns)}  rows: {control_test.shape[0]}")

# %% === Step 5: Combine train/test ===
amyloid_train['group'] = 'amyloid'
control_train['group'] = 'control'
pyp_negatives_train['group'] = 'pyp_negative'
train_final = pd.concat([control_train, amyloid_train, pyp_negatives_train],
                        ignore_index=True)

amyloid_test['group'] = 'amyloid'
control_test['group'] = 'control'
pyp_negatives_test['group'] = 'pyp_negative'
test_final = pd.concat([amyloid_test, control_test, pyp_negatives_test],
                       ignore_index=True)

# Drop unknown-age rows; one ECG per FileID
train_final.dropna(subset='Age', inplace=True)
test_final.dropna(subset='Age', inplace=True)
test_final = test_final.drop_duplicates(subset='FileID', keep='first')
train_final = train_final.drop_duplicates(subset='FileID')

# %% === Step 6: Save ===
train_final.to_csv(os.path.join(project.tabs_path, 'cohort_train.csv'), index=False)
test_final.to_csv(os.path.join(project.tabs_path, 'cohort_test.csv'), index=False)

train_mrns = set(train_final['MRN'])
test_mrns = set(test_final['MRN'])
overlap = train_mrns & test_mrns
print(f"\n🚫 Train/test MRN overlap: {len(overlap)}")
if overlap:
    print("Sample overlapping MRNs:", list(overlap)[:20])
    raise RuntimeError(f"Train/test MRN overlap found: {len(overlap)} MRNs — aborting.")

print("\n✅ Final cohort sizes:")
print(f"  • Train MRNs: {train_final['MRN'].nunique()}  ECGs: {len(train_final)}")
print(f"  • Test  MRNs: {test_final['MRN'].nunique()}  ECGs: {len(test_final)}")


# %% === Step 7: Summaries ===
def summarize_groups(df, label):
    print(f"\n--- {label} ---")
    print(f"Total ECG rows: {len(df):,}    Unique MRNs: {df['MRN'].nunique():,}")
    rows_by_group = df.groupby('group').size().rename('ecg_rows')
    mrns_by_group = df.groupby('group')['MRN'].nunique().rename('unique_mrns')
    summary = pd.concat([rows_by_group, mrns_by_group], axis=1).reset_index()
    print(summary.to_string(index=False))
    return summary


summarize_groups(train_final, "TRAIN")
summarize_groups(test_final, "TEST")
# %%
