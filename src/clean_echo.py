"""
Echo phenotype derivation (LVH / aortic stenosis) + amyloid ICD flag.

Ported from multimodal_amyloid/code/clean_echodata2.py.

NOTE: A PyP-only data update does NOT change echo or ICD data, so the standard
pipeline REUSES the research repo's amyloid_echo_outcomes.csv
(project.echo_outcomes_file) and does not run this script. It is kept here for
reproducibility / if the echo extract is ever refreshed. The ICD multiprocessing
block writes to THIS repo's tabs; repoint project.echo_outcomes_file there if you
regenerate it.
"""
# %%
import os
from functools import reduce

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import project_constants as project


# === Echo field cleaning ===
def clean_ivsd(x):
    if pd.isna(x) or x < 0.4 or (4 < x < 5):  # remove likely errors
        return np.nan
    elif x >= 5:
        return x / 10  # mm -> cm
    return x


def composite_lvh_binary(row):
    ivsd_crit = pd.notna(row['IVSd_cleaned']) and row['IVSd_cleaned'] > 1.4
    wall_crit = row['LVWallThickness_cleaned'] in ['moderately increased', 'severely increased']
    if ivsd_crit or wall_crit:
        return 1
    elif pd.notna(row['IVSd_cleaned']) or pd.notna(row['LVWallThickness_cleaned']):
        return 0
    return np.nan


def AS_Objective(row):
    if row['AVPkVel(m/s)'] >= 4 or row['AVMnGrad(mmHg)'] >= 40 or row['AVAContVTI'] < 1 or row['AVAIndex'] < 0.6:
        return 'Severe'
    elif 3 <= row['AVPkVel(m/s)'] < 4:
        return 'Moderate'
    elif 20 <= row['AVMnGrad(mmHg)'] < 40:
        return 'Moderate'
    elif 1 <= row['AVAContVTI'] <= 1.5:
        return 'Moderate'
    elif 0.6 <= row['AVAIndex'] <= 0.85:
        return 'Moderate'
    elif row['AVPkVel(m/s)'] < 3 or row['AVMnGrad(mmHg)'] < 20 or row['AVAContVTI'] > 1.5 or row['AVAIndex'] > 0.85:
        return 'Mild'
    return np.nan


def severe_as_objsubj(row):
    if row['AS_Objective'] == 'Severe':
        return True
    elif row['AS_Objective'] in ('Moderate', 'Mild'):
        return False
    elif row['AVStenosis'] == 'Severe':
        return True
    elif pd.isna(row['AVStenosis']):
        return np.nan
    return False


def moderate_as_objsubj(row):
    if row['AS_Objective'] == 'Moderate':
        return True
    elif row['AS_Objective'] in ('Severe', 'Mild'):
        return False
    elif row['AVStenosis'] in ('Moderate', 'Mod-Sev', 'Mild-Mod'):
        return True
    elif pd.isna(row['AVStenosis']):
        return np.nan
    return False


# === Load + derive phenotypes ===
echo = pd.read_csv(
    os.path.join(project.signals,
                 'data_LD/data_feb2025/echodata_composite_flagged_till06Feb2025.csv'),
    low_memory=False)
echo['PAT_MRN_ID'] = echo['MRN']
echo['IVSd_cleaned'] = echo['IVSd'].apply(clean_ivsd)
echo['LVWallThickness_cleaned'] = echo['LVWallThickness'].astype(str).str.strip().str.lower()
echo['Composite_LVH_binary'] = echo.apply(composite_lvh_binary, axis=1)
echo['AS_Objective'] = echo.apply(AS_Objective, axis=1)
echo['ModerateAS_ObjSubj'] = echo.apply(moderate_as_objsubj, axis=1)
echo['SevereAS_ObjSubj'] = echo.apply(severe_as_objsubj, axis=1)

print("\nComposite LVH distribution:")
print(echo['Composite_LVH_binary'].value_counts(dropna=False))
print("\nSevere AS distribution:")
print(echo['SevereAS_ObjSubj'].value_counts(dropna=False))

echo.to_csv(os.path.join(project.tabs_path, 'echo_cleaned.csv'), index=False)

# %% === Amyloid ICD (E85) flag via parallel diagnosis-file scan ===
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

MRNs = echo['MRN'].unique()
chunk_size = 100000
n_patients = len(MRNs)
dx_df_cols = ['PAT_ID', 'PAT_MRN_ID', 'DX_SOURCE', 'DX_DATE', 'CURRENT_ICD10_LIST']
all_icd_prefixes = ["E85"]
filenames = ['CarDS_2435227_Hosp_Enc_DX.txt', 'CarDS_2435227_Outpatient_Enc_DX.txt']
dx_files = [os.path.join(project.s3_implementation, name) for name in filenames]


def process_icd_chunk(start_idx, MRNs, chunk_size, dx_files, dx_df_cols, prefixes):
    import project_constants as project
    import pandas as pd
    import os, time
    start = time.time()
    end_idx = min(start_idx + chunk_size, len(MRNs))
    current_chunk = MRNs[start_idx:end_idx]
    dx_chunks = []
    for f in dx_files:
        dx_chunks.append(project.load_filtered_data(
            f, chunk_size=10**8, columns=dx_df_cols, sep='\t', incl_MRNs=current_chunk))
    dx = pd.concat(dx_chunks, ignore_index=True)
    dx.dropna(subset=['DX_DATE', 'CURRENT_ICD10_LIST'], inplace=True)
    dx['DX_DATE'] = pd.to_datetime(dx['DX_DATE'], errors='coerce')
    dx = dx.dropna(subset=['DX_DATE'])
    dx['CURRENT_ICD10_LIST'] = dx['CURRENT_ICD10_LIST'].str.split(', ')
    dx = dx.explode('CURRENT_ICD10_LIST')
    dx = dx[dx['CURRENT_ICD10_LIST'].astype(str).str.startswith(tuple(prefixes))]
    print(f"[PID {os.getpid()}] chunk {start_idx} done in {time.time() - start:.1f}s", flush=True)
    return dx


def assign_binary_outcome_from_icd(df_long, icd_prefixes, outcome_name):
    df = df_long[['PAT_MRN_ID', 'CURRENT_ICD10_LIST']].dropna().copy()
    icd_mask = df['CURRENT_ICD10_LIST'].astype(str).str.startswith(tuple(icd_prefixes))
    matched = df.loc[icd_mask, 'PAT_MRN_ID'].unique()
    out = pd.DataFrame({'PAT_MRN_ID': df['PAT_MRN_ID'].unique()})
    out[outcome_name] = out['PAT_MRN_ID'].isin(matched).astype(int)
    return out


if __name__ == "__main__":
    max_workers = max(1, int(os.cpu_count() / 4))
    print(f"Max workers: {max_workers}")
    all_dx = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_icd_chunk, i, MRNs, chunk_size, dx_files,
                             dx_df_cols, all_icd_prefixes)
                   for i in range(0, n_patients, chunk_size)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="MRN chunks"):
            try:
                all_dx.append(f.result())
            except Exception as e:
                print(f"Error in chunk: {e}")

    icd_long = pd.concat(all_dx, ignore_index=True)
    phenotypes = {'Amyloid_ICD': ['E85']}
    binary_dfs = [assign_binary_outcome_from_icd(echo.merge(icd_long, on='PAT_MRN_ID', how='left'),
                                                 prefixes, name)
                  for name, prefixes in phenotypes.items()]
    outcomes_df = reduce(lambda l, r: pd.merge(l, r, on='PAT_MRN_ID'), binary_dfs)
    final_echo_df = echo.drop_duplicates().merge(outcomes_df, on=['PAT_MRN_ID'], how='left')
    out = os.path.join(project.tabs_path, 'amyloid_echo_outcomes.csv')
    final_echo_df.to_csv(out, index=False)
    print(f"✅ Saved {out}")
