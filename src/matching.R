# %% === Setup ===
# Ported from multimodal_amyloid/code/matching1_10_10.R
# 1:20 difficulty-enriched matching on Age + Sex (patient level), two arms:
#   amyloid vs control   AND   amyloid vs lvh
# The amyloid-vs-lvh arm is what makes the negatives hard.
#
# Tuning dials for "more training data" (change with counts visible):
#   - ratio (project MATCH_RATIO = 20)
#   - slice_head(n = 5): ECGs per patient cap
#   - pyp_negative group is dropped here (only amyloid/control/lvh retained)
.libPaths(c("~/R/library", .libPaths()))
library(MatchIt)
library(dplyr)
library(readr)

set.seed(20250910)

RATIO          <- 20
MAX_ECG        <- 5
COHORT_TRAIN   <- "/home/pmc57/projects/train_ECGVISION_ATTR/tabs/cohort_train.csv"
OUT_PATH       <- "/home/pmc57/projects/train_ECGVISION_ATTR/tabs/train_matched_1_20.csv"

# === Load data ===
data <- read_csv(COHORT_TRAIN)
data$PatientSex_ECGData <- factor(data$PatientSex_ECGData)

# === Collapse to patient level ===
patients <- data %>%
  group_by(MRN) %>%
  summarise(
    Age = first(Age),
    PatientSex_ECGData = first(PatientSex_ECGData),
    group = first(group),
    Composite_LVH_binary = max(Composite_LVH_binary, na.rm = TRUE),
    SevereAS_ObjSubj = max(SevereAS_ObjSubj, na.rm = TRUE),
    .groups = "drop"
  )

# === Define groups (patient level) ===
amyloid_pat <- patients %>% filter(group == "amyloid")
lvh_pat <- patients %>%
  filter(Composite_LVH_binary == 1 | SevereAS_ObjSubj == 1) %>%
  mutate(group = "lvh")
control_pat <- patients %>% filter(group == "control")

# === Matching function (patient level) ===
run_matching <- function(case_data, pool_data, ratio = RATIO, label = "match") {
  match_input <- bind_rows(
    case_data %>% mutate(treat = 1),
    pool_data %>% mutate(treat = 0)
  )
  m.out <- matchit(
    treat ~ Age + PatientSex_ECGData,
    data = match_input,
    method = "nearest",
    ratio = ratio
  )
  matched_df <- match.data(m.out)
  matched_df$match_group <- label
  return(matched_df)
}

# === Run matchings ===
amyloid_vs_control <- run_matching(amyloid_pat, control_pat, ratio = RATIO, label = "amyloid_control")
amyloid_vs_lvh     <- run_matching(amyloid_pat, lvh_pat,     ratio = RATIO, label = "amyloid_lvh")

# === Extract groups (patients) ===
amyloid_cases_pat    <- amyloid_vs_control %>% filter(treat == 1)
controls_matched_pat <- amyloid_vs_control %>% filter(treat == 0) %>% distinct(MRN, .keep_all = TRUE)
lvh_matched_pat <- amyloid_vs_lvh %>%
  filter(treat == 0 & !(MRN %in% amyloid_cases_pat$MRN)) %>%
  distinct(MRN, .keep_all = TRUE)

all_matched_pat <- bind_rows(amyloid_cases_pat, controls_matched_pat, lvh_matched_pat)

# === Expand back to ECG level (cap MAX_ECG per patient, most recent first) ===
all_matched_ecg <- data %>%
  filter(MRN %in% all_matched_pat$MRN) %>%
  arrange(MRN, desc(ECGDate)) %>%
  group_by(MRN) %>%
  slice_head(n = MAX_ECG) %>%
  ungroup()

# === Clean group label ===
final_matched <- all_matched_ecg %>%
  mutate(group = case_when(
    MRN %in% amyloid_cases_pat$MRN ~ "amyloid",
    MRN %in% controls_matched_pat$MRN ~ "control",
    MRN %in% lvh_matched_pat$MRN ~ "lvh"
  ))

# === Save ===
write_csv(final_matched, OUT_PATH)
cat("✅ Final matched cohort saved to", OUT_PATH, "\n\n")

cat("Counts (patients and ECGs):\n")
print(
  final_matched %>%
    group_by(group) %>%
    summarise(patients = n_distinct(MRN), ecgs = n(), .groups = "drop")
)

cat("\nBalance check (Age and Sex by group):\n")
print(
  final_matched %>%
    group_by(group) %>%
    summarise(
      mean_age = mean(Age, na.rm = TRUE),
      sd_age   = sd(Age, na.rm = TRUE),
      pct_male = mean(PatientSex_ECGData == "M", na.rm = TRUE) * 100,
      .groups = "drop"
    )
)
