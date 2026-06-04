"""
Quick comparison of medication files across the two data releases.
Run on the server: python check_med_files.py
"""
import os
import pandas as pd

S3 = os.path.expanduser(
    '~/s3_implementation/cardsjdat-CC1022-MEDINT/2435227-CarDS-ECG')

FILES = {
    '2025 Med_Admin_1': os.path.join(S3, 'Data-2025-04-03', '2435227_CarDS_ECG_Med_Admin_1.txt'),
    '2025 Med_Admin_2': os.path.join(S3, 'Data-2025-04-03', '2435227_CarDS_ECG_Med_Admin_2.txt'),
    '2025 Med_Admin_3': os.path.join(S3, 'Data-2025-04-03', '2435227_CarDS_ECG_Med_Admin_3.txt'),
    '2025 Meds_1':      os.path.join(S3, 'Data-2025-04-03', '2435227_CarDS_ECG_Meds_1.txt'),
    '2025 Meds_2':      os.path.join(S3, 'Data-2025-04-03', '2435227_CarDS_ECG_Meds_2.txt'),
    '2026 Meds':        os.path.join(S3, 'Data-2026-04-15', 'CarDS_2435227_Meds.txt'),
    '2026 Outpt Med':   os.path.join(S3, 'Data-2026-04-15', 'CarDS_2435227_Outpatient_Enc_Med_Admin.txt'),
}

DATE_COLS = ['ORDER_DATE', 'ADMIN_DATE', 'ADMINISTERED_DATE', 'ORDER_INST', 'DATE']

for label, path in FILES.items():
    if not os.path.exists(path):
        print(f"{label:25s}  NOT FOUND")
        continue

    size_mb = os.path.getsize(path) / 1e6
    df = pd.read_csv(path, sep='\t', nrows=5000, dtype=str,
                     on_bad_lines='skip', low_memory=False)
    n_rows_sample = len(df)
    n_mrns = df['PAT_MRN_ID'].nunique() if 'PAT_MRN_ID' in df.columns else '?'

    # date range from sample
    date_min, date_max = None, None
    for dc in DATE_COLS:
        if dc in df.columns:
            parsed = pd.to_datetime(df[dc], errors='coerce')
            if parsed.notna().any():
                date_min = parsed.min()
                date_max = parsed.max()
                break

    # full row count (fast)
    with open(path, 'rb') as f:
        total_rows = sum(1 for _ in f) - 1  # subtract header

    print(f"{label:25s}  {size_mb:7.1f} MB  {total_rows:>10,} rows  "
          f"MRNs(sample)={n_mrns}  "
          f"dates(sample): {date_min.date() if date_min else '?'} → {date_max.date() if date_max else '?'}")
