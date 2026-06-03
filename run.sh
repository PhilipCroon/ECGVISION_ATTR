#!/bin/bash
set -e

CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$CODE_DIR/output.txt"

echo "=== ECGVISION-ATTR data prep ===" > "$LOG_FILE"
date >> "$LOG_FILE"

# Echo phenotypes are REUSED from the research repo by default — clean_echo.py is
# only needed if the echo extract was refreshed. Uncomment to regenerate:
# echo -e "\n--- clean_echo.py ---" >> "$LOG_FILE"
# python "$CODE_DIR/src/clean_echo.py" >> "$LOG_FILE" 2>&1

echo -e "\n--- build_cohort.py ---" >> "$LOG_FILE"
python "$CODE_DIR/src/build_cohort.py" >> "$LOG_FILE" 2>&1

echo -e "\n--- matching.R ---" >> "$LOG_FILE"
Rscript "$CODE_DIR/src/matching.R" >> "$LOG_FILE" 2>&1

echo -e "\n✅ Data prep complete. Train with: python train/train.py" >> "$LOG_FILE"
