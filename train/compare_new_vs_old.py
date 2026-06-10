"""
One-shot: compare the new ECGVISION-ATTR model vs the old amyloid model on test.
Both paths baked in — just run it:

    python compare_new_vs_old.py

NEW = attr_amyloid_2026_06_05_unfrozen_06 (best unfrozen checkpoint, val_auroc=0.807).
OLD = trained_model_Amyloidosis_stage2_age_sex_1_10_15 (epoch 15, image-only).
Both expect 300x300x3, [0,1]-scaled input — apples-to-apples on cohort_test.csv.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import project_constants as project

MODEL_DIR = os.path.join(project.project_root, 'models')
NEW_MODEL = os.path.join(MODEL_DIR, 'attr_amyloid_2026_06_05_unfrozen_06')
OLD_MODEL = '/home/pmc57/cmp-jdat-data/variant_amyloid/models/trained_model_Amyloidosis_stage2_age_sex_1_10_15'


def main():
    new_model = NEW_MODEL
    print(f"NEW: {new_model}")
    print(f"OLD: {OLD_MODEL}")
    assert os.path.exists(new_model), (
        f"NEW model not found at {new_model} — check the dir name in models/")
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
