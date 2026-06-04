"""
Baseline characteristics table for the ECGVISION-ATTR training cohort.

Reads cohort_train.csv and cohort_test.csv (produced by build_cohort.py) and
the matched training cohort (train_matched_1_20.csv from matching.R).

Outputs: tabs/baseline_table.xlsx  (one sheet per cohort stage)

Run after build_cohort.py (and optionally matching.R).
"""
# %%
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project

OUT_XLSX = os.path.join(project.tabs_path, 'baseline_table.xlsx')

# ---------- helpers ----------

def clean_sex(x):
    x = str(x).strip().upper()
    if x in ('F', 'FEMALE'):
        return 'F'
    if x in ('M', 'MALE'):
        return 'M'
    return np.nan


def fmt_n_pct(n, total):
    return f"{int(n)} ({n / total * 100:.1f}%)" if total > 0 else "0"


def baseline_group(df_patients, group_label):
    """One-column baseline for a patient-level DataFrame."""
    n = len(df_patients)
    if n == 0:
        return pd.Series({'N': 0}, name=group_label)

    age = pd.to_numeric(df_patients.get('Age', pd.Series(dtype=float)), errors='coerce')
    sex_col = next((c for c in ('PatientSex_ECGData', 'Sex', 'SEX') if c in df_patients.columns), None)
    sex = df_patients[sex_col].apply(clean_sex) if sex_col else pd.Series(dtype=str)

    stats = {
        'N': str(n),
        'Age mean ± SD': f"{age.mean():.1f} ± {age.std():.1f}" if age.notna().any() else 'N/A',
        'Female n (%)': fmt_n_pct((sex == 'F').sum(), n),
        'Male n (%)': fmt_n_pct((sex == 'M').sum(), n),
    }

    for col, label in [
        ('Composite_LVH_binary', 'LVH n (%)'),
        ('SevereAS_ObjSubj', 'Severe AS n (%)'),
    ]:
        if col in df_patients.columns:
            vals = pd.to_numeric(df_patients[col], errors='coerce').fillna(0)
            stats[label] = fmt_n_pct(int(vals.sum()), n)

    return pd.Series(stats, name=group_label)


def make_table(df_ecg, label):
    """
    Build a multi-group baseline table from an ECG-level DataFrame.
    Deduplicates to first ECG per MRN before computing stats.
    """
    df_ecg = df_ecg.copy()
    df_ecg['ECGDate'] = pd.to_datetime(df_ecg.get('ECGDate'), errors='coerce')
    df_pat = (df_ecg.sort_values('ECGDate')
              .groupby('MRN', as_index=False)
              .first())

    groups = sorted(df_pat['group'].dropna().unique())
    cols = [baseline_group(df_pat, 'Overall')]
    for g in groups:
        cols.append(baseline_group(df_pat[df_pat['group'] == g], g.capitalize()))

    tbl = pd.concat(cols, axis=1)
    print(f"\n=== {label} ===")
    print(tbl.to_string())
    return tbl


# ---------- main ----------
tables = {}

for fname, label in [
    ('cohort_train.csv', 'Train (pre-matching)'),
    ('cohort_test.csv', 'Test (temporal holdout)'),
    ('train_matched_1_20.csv', 'Train matched 1:20'),
]:
    path = os.path.join(project.tabs_path, fname)
    if not os.path.exists(path):
        print(f"Skipping {label}: {path} not found")
        continue
    df = pd.read_csv(path, low_memory=False)
    if 'group' not in df.columns:
        print(f"Skipping {label}: no 'group' column in {fname}")
        continue
    tables[label] = make_table(df, label)

if not tables:
    raise SystemExit("No cohort files found. Run build_cohort.py first.")

with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as writer:
    for sheet_name, tbl in tables.items():
        safe_name = sheet_name[:31]  # Excel sheet name limit
        tbl.to_excel(writer, sheet_name=safe_name)

print(f"\nSaved: {OUT_XLSX}")
# %%
