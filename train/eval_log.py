"""
Shared evaluation log. Every test script appends one row per (model, cohort) to
models/eval_log.csv, so a checkpoint's val/internal/external numbers live in one
place — joinable to models/run_registry.csv on the checkpoint name (best_ckpt).

Usage:
    from eval_log import log_eval, EVAL_LOG
    log_eval(model_path, cohort='WVU', render='deploy', metrics={'auroc':..., ...})
"""
import os
import csv
import subprocess
from datetime import datetime

import project_constants as project

EVAL_LOG = os.path.join(project.project_root, 'models', 'eval_log.csv')
FIELDS = ['ts', 'git_sha', 'model', 'cohort', 'render', 'level',
          'n', 'pos', 'auroc', 'auprc', 'sens_90spec', 'spec_90sens', 'notes']


def _git_sha():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


def log_eval(model, cohort, metrics, render='', level='patient', notes=''):
    """Append one evaluation row. `model` may be a path (basename is stored)."""
    row = {
        'ts': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'git_sha': _git_sha(),
        'model': os.path.basename(str(model)),
        'cohort': cohort,
        'render': render,
        'level': level,
        'notes': notes,
    }
    for k in ('n', 'pos', 'auroc', 'auprc', 'sens_90spec', 'spec_90sens'):
        v = metrics.get(k)
        row[k] = round(v, 5) if isinstance(v, float) else v
    os.makedirs(os.path.dirname(EVAL_LOG), exist_ok=True)
    write_header = not os.path.exists(EVAL_LOG)
    with open(EVAL_LOG, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow(row)
