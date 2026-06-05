"""
One-shot: compare the new ECGVISION-ATTR model vs the old amyloid model on test.
Both paths baked in — just run it:

    python compare_new_vs_old.py

NEW = best unfrozen checkpoint from the latest training run (auto-detected).
OLD = trained_model_Amyloidosis_stage2_age_sex_1_10_15 (epoch 15, image-only).
Both expect 300x300x3, [0,1]-scaled input — apples-to-apples on cohort_test.csv.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project

MODEL_DIR = os.path.join(project.project_root, 'models')
OLD_MODEL = '/home/pmc57/cmp-jdat-data/variant_amyloid/models/trained_model_Amyloidosis_stage2_age_sex_1_10_15'


def latest_new_checkpoint():
    entries = [
        os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR)
        if 'unfrozen' in d and os.path.isdir(os.path.join(MODEL_DIR, d))
    ]
    if not entries:
        raise FileNotFoundError(f"No unfrozen checkpoint in {MODEL_DIR}")
    return max(entries, key=os.path.getmtime)


def main():
    new_model = latest_new_checkpoint()
    print(f"NEW: {new_model}")
    print(f"OLD: {OLD_MODEL}")
    if not os.path.exists(OLD_MODEL):
        print(f"\nWARNING: old model not found at {OLD_MODEL} — running NEW only.")
        models = [new_model]
    else:
        models = [new_model, OLD_MODEL]

    # Hand off to the comparison engine
    sys.argv = ['compare_on_test.py'] + models
    import compare_on_test
    compare_on_test.main()


if __name__ == '__main__':
    main()
