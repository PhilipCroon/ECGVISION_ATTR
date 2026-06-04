"""
Search 2026 medication files for ATTR-specific drugs and produce a deduplicated
MRN-level file with the earliest medication start date.

Output:
  tabs/tafamidis_first_by_mrn.csv  (MRN, first_med_date, first_med_source, first_med_name, match_count)

Notes:
  - Uses only Data-2026-04-15 (full historical refresh back to 2011 — 2025 files subset of this).
  - Searches both inpatient (Meds) and outpatient (Outpatient_Enc_Med_Admin) files.
  - Deduplication: keeps earliest date per MRN across all files.
"""
# %%
import os
import re
import sys

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project

# ---------- CONFIG ----------
S3_2026 = os.path.join(
    project.s3_implementation.replace('Data-2025-04-03', 'Data-2026-04-15'))

INPUT_FILES = [
    os.path.join(S3_2026, 'CarDS_2435227_Meds.txt'),
    os.path.join(S3_2026, 'CarDS_2435227_Outpatient_Enc_Med_Admin.txt'),
]

OUT_DEDUP = os.path.join(project.tabs_path, 'tafamidis_first_by_mrn.csv')
CHUNKSIZE = 200_000

# All ATTR-specific drugs
KEYWORDS = [
    'tafamidis', 'vyndaqel', 'vyndamax',
    'acoramidis', 'attruby',
    'vutrisiran', 'amvuttra',
    'patisiran', 'onpattro',
    'inotersen', 'tegsedi',
    'eplontersen', 'wainua',
]
pattern_str = r'(?i)\b(?:' + r'|'.join(re.escape(k) for k in KEYWORDS) + r')\b'

DATE_COL_CANDIDATES = ['ORDER_DATE', 'ADMIN_DATE', 'ADMINISTERED_DATE', 'ORDER_INST', 'DATE']

# ---------- helpers ----------
def try_read_header(fn):
    for sep in ['\t', ',']:
        try:
            cols = pd.read_csv(fn, sep=sep, nrows=0).columns.tolist()
            return cols, sep
        except Exception:
            continue
    cols = pd.read_csv(fn, sep=None, engine='python', nrows=0).columns.tolist()
    return cols, None


def choose_usecols(cols):
    usecols = []
    for cand in ['PAT_MRN_ID', 'MRN', 'PAT_ID']:
        if cand in cols:
            usecols.append(cand)
            break
    for cand in ['MEDICATION_NAME', 'GENERIC_NAME', 'MED_NAME', 'MED', 'Meds', 'MEDICATION']:
        if cand in cols:
            usecols.append(cand)
    med_like = [c for c in cols
                if any(tok in c.lower() for tok in ('med', 'drug', 'generic'))
                and c not in usecols]
    usecols.extend(med_like)
    for cand in DATE_COL_CANDIDATES:
        if cand in cols and cand not in usecols:
            usecols.append(cand)
    return usecols if usecols else None


def get_med_cols(cols):
    exact = {'medication_name', 'generic_name', 'med_name', 'med', 'meds', 'medication'}
    meds = [c for c in cols if c.lower() in exact]
    if meds:
        return meds
    return [c for c in cols
            if any(tok in c.lower() for tok in ('med', 'drug', 'generic'))]


# ---------- main ----------
for p in INPUT_FILES:
    if not os.path.exists(p):
        print(f"WARNING: not found — {p}")

file_paths = [p for p in INPUT_FILES if os.path.exists(p)]
if not file_paths:
    raise SystemExit("No input files found. Check s3_implementation path in project_constants.py")

# mrn -> {date, source, med_name, count}
mrn_map = {}

print("Scanning files...")
for fn in tqdm(file_paths):
    try:
        cols, sep = try_read_header(fn)
    except Exception as e:
        print(f"Could not read header for {fn} — skipping: {e}")
        continue

    usecols = choose_usecols(cols)
    med_cols = [c for c in get_med_cols(cols) if c in cols]
    date_cols = [c for c in DATE_COL_CANDIDATES if c in cols]

    sep_list = [sep] if sep else ['\t', ',']
    iterator = None
    for s in sep_list:
        try:
            iterator = pd.read_csv(fn, sep=s, usecols=usecols, dtype=str,
                                   chunksize=CHUNKSIZE, on_bad_lines='skip',
                                   low_memory=False, encoding='utf-8')
            break
        except Exception as e:
            last_exc = e
    if iterator is None:
        print(f"ERROR opening {fn}: {last_exc}")
        continue

    for chunk in iterator:
        chunk.columns = [c.strip() for c in chunk.columns]

        med_cols_chunk = [c for c in med_cols if c in chunk.columns]
        if not med_cols_chunk:
            med_cols_chunk = [c for c in chunk.columns if chunk[c].dtype == object]

        masks = []
        for c in med_cols_chunk:
            s = chunk[c].fillna('').astype(str)
            try:
                masks.append(s.str.contains(pattern_str, regex=True, na=False))
            except Exception:
                masks.append(s.str.lower().str.contains(
                    '|'.join(k.lower() for k in KEYWORDS), na=False))

        if not masks:
            continue
        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m
        if not combined_mask.any():
            continue

        matched = chunk.loc[combined_mask].copy()

        mrn_col = next((c for c in ['PAT_MRN_ID', 'MRN', 'PAT_ID']
                        if c in matched.columns), None)
        if mrn_col is None:
            print(f"WARNING: no MRN column in {fn} — skipping matched rows")
            continue

        available_date_cols = [c for c in date_cols if c in matched.columns]
        if available_date_cols:
            ds = matched[available_date_cols].astype(str).replace('nan', '').replace('None', '')
            first_date_str = ds.replace(r'^\s*$', pd.NA, regex=True).bfill(axis=1).iloc[:, 0]
            first_date = pd.to_datetime(first_date_str, errors='coerce')
        else:
            first_date = pd.Series([pd.NaT] * len(matched), index=matched.index)

        name_col = next((c for c in ['MEDICATION_NAME', 'GENERIC_NAME', 'MED_NAME']
                         if c in matched.columns), None)
        med_name_series = (matched[name_col].astype(str).fillna('')
                           if name_col else
                           pd.Series([''] * len(matched), index=matched.index))

        for idx, row in matched.iterrows():
            mrn = row[mrn_col]
            if pd.isna(mrn) or str(mrn).strip() == '':
                continue

            dt = first_date.loc[idx] if idx in first_date.index else pd.NaT
            if pd.isna(dt) and 'ORDER_INST' in matched.columns:
                try:
                    dt = pd.to_datetime(str(matched.loc[idx, 'ORDER_INST']), errors='coerce')
                except Exception:
                    pass

            mednm = str(med_name_series.loc[idx]).strip() if idx in med_name_series.index else ''
            if mednm in ('nan', 'None'):
                mednm = ''

            cur = mrn_map.get(mrn)
            if cur is None:
                mrn_map[mrn] = {
                    'date': pd.NaT if pd.isna(dt) else pd.Timestamp(dt),
                    'source': os.path.basename(fn),
                    'med_name': mednm,
                    'count': 1,
                }
            else:
                cur['count'] += 1
                if not pd.isna(dt):
                    dt_ts = pd.Timestamp(dt)
                    if pd.isna(cur['date']) or dt_ts < cur['date']:
                        cur['date'] = dt_ts
                        cur['source'] = os.path.basename(fn)
                        cur['med_name'] = mednm

# ---------- output ----------
rows = [
    {
        'MRN': mrn,
        'first_med_date': info['date'].isoformat() if not pd.isna(info['date']) else '',
        'first_med_source': info['source'],
        'first_med_name': info['med_name'],
        'match_count': info['count'],
    }
    for mrn, info in mrn_map.items()
]

df_out = pd.DataFrame(rows, columns=['MRN', 'first_med_date', 'first_med_source',
                                     'first_med_name', 'match_count'])
df_out['_dt'] = pd.to_datetime(df_out['first_med_date'], errors='coerce')
df_out = df_out.sort_values(['_dt', 'MRN']).drop(columns=['_dt'])
df_out.to_csv(OUT_DEDUP, index=False)

print(f"Wrote: {OUT_DEDUP}")
print(f"Unique MRNs: {df_out.shape[0]}")
# %%
