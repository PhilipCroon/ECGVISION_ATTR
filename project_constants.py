"""
Project paths and shared constants for train_ECGVISION_ATTR.

Mirrors multimodal_amyloid/code/project_constants.py but:
  - writes outputs to this repo's own tabs/ (does not touch the research repo)
  - reads upstream artifacts (echo outcomes, ECG metadata, demographics)
    from the research repo, since a PyP-only data update leaves them unchanged
  - points at the new PyP parquet (March-2026 update)

All paths target the Yale compute box.
"""
import os

# === Raw data sources (shared, read-only) ===
data_path = '/home/pmc57/cmp-jdat-data/variant_amyloid'
signals = '/mnt/nfs_yale_ecg_signals'
s3_implementation = '/home/pmc57/s3_implementation/cardsjdat-CC1022-MEDINT/2435227-CarDS-ECG/Data-2025-04-03'

# === Upstream artifacts reused from the research repo (unchanged by PyP update) ===
source_tabs = '/home/pmc57/projects/multimodal_amyloid/tabs'

# Reused files (read-only). Confirm these did NOT also update before running.
echo_outcomes_file = os.path.join(source_tabs, 'amyloid_echo_outcomes.csv')
ecg_metadata_file = os.path.join(
    signals, 'data_LD/ecg_metadata_flagged_01jan2000_to_17july2025.csv')
tafamidis_file = os.path.join(source_tabs, 'tafamidis_first_by_mrn.csv')

# === New PyP data update (March 2026) ===
pyp_file = os.path.join(data_path, 'amyloid_pyp_all_03102026.parquet')

# === This repo's outputs ===
project_root = '/home/pmc57/projects/train_ECGVISION_ATTR'
tabs_path = os.path.join(project_root, 'tabs')
final_files_path = os.path.join(project_root, 'final_files')
os.makedirs(tabs_path, exist_ok=True)
os.makedirs(final_files_path, exist_ok=True)

# === Cohort split ===
# Later than the research repo (2024-07-01): external validation covers
# generalization, so the internal test is a small recent temporal holdout and
# the bulk of patients go to training.
CUTOFF_DATE = '2025-07-01'

# === Matching ===
MATCH_RATIO = 20            # 1:20 difficulty enrichment (amyloid vs control AND vs lvh)
MAX_ECG_PER_PATIENT = 5     # ECGs kept per matched patient


# === Training-specific paths (change these when moving servers) ===
# utils.py and model_helpers.py live in this repo — ecg_image_models_path = project_root
ecg_image_models_path = project_root
signals_base = '/mnt/raid0/bb2238/signals'   # raw ECG .npy root (numpy_rp/ and numpy/ subdirs)
contrastive_weights = '/mnt/nfs_yale_ecg/contrastive_pre2015/B3/B3_biocontrastive_epoch10'
image_dir = '/home/pmc57/precomputed_images_attr'   # writable dir — created on first MAKE_IMAGE run
formats_file = '/mnt/nfs_yale_ecg/preprocessing/formats_rerun.csv'

# === Chunked CSV loaders (used by clean_echo.py ICD scan) ===
def load_data_in_chunks(file_path, columns=None, chunk_size=10000, sep=',', n_rows=None):
    import pandas as pd
    return pd.read_csv(
        file_path, usecols=columns, chunksize=chunk_size, memory_map=True,
        low_memory=False, on_bad_lines='skip', sep=sep, nrows=n_rows)


def load_filtered_data(file_path, chunk_size, columns, sep, incl_MRNs, nrows=None):
    import pandas as pd
    filtered = []
    for chunk in load_data_in_chunks(file_path, chunk_size=chunk_size, columns=columns,
                                     sep=sep, n_rows=nrows):
        filtered.append(chunk[chunk['PAT_MRN_ID'].isin(incl_MRNs)])
    return pd.concat(filtered)
