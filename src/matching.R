# %% === Setup ===
# Simplified 1:10 age+sex matching (replaces the previous 1:20 two-arm
# control+lvh matching). Single arm: amyloid vs control, nearest-neighbour on
# Age + Sex. Controls capped at 5 most-recent ECGs/patient; amyloid uncapped
# (more positive signal, since amyloid is the rare class). pyp_negative is not
# used as a training negative.
.libPaths(c("~/R/library", .libPaths()))
library(MatchIt)
library(dplyr)
library(readr)

set.seed(20250910)

RATIO        <- 10
CONTROL_MAX_ECG <- 5
COHORT_TRAIN <- "/home/pmc57/projects/train_ECGVISION_ATTR/tabs/cohort_train.csv"
OUT_PATH     <- "/home/pmc57/projects/train_ECGVISION_ATTR/tabs/train_matched_1_10.csv"

# === Load data ===
data <- read_csv(COHORT_TRAIN)
data$PatientSex_ECGData <- factor(data$PatientSex_ECGData)

# === Collapse to patient level (Age + Sex for matching) ===
patients <- data %>%
  group_by(MRN) %>%
  summarise(
    Age = first(Age),
    PatientSex_ECGData = first(PatientSex_ECGData),
    group = first(group),
    .groups = "drop"
  )

amyloid_pat <- patients %>% filter(group == "amyloid")
control_pat <- patients %>% filter(group == "control")

cat(sprintf("Amyloid patients: %d   Control pool: %d\n",
            nrow(amyloid_pat), nrow(control_pat)))

# === 1:RATIO nearest-neighbour match on Age + Sex (patient level) ===
match_input <- bind_rows(
  amyloid_pat %>% mutate(treat = 1),
  control_pat %>% mutate(treat = 0)
)
m.out <- matchit(
  treat ~ Age + PatientSex_ECGData,
  data = match_input,
  method = "nearest",
  ratio = RATIO
)
matched <- match.data(m.out)

amyloid_cases_pat    <- matched %>% filter(treat == 1)
controls_matched_pat <- matched %>% filter(treat == 0) %>% distinct(MRN, .keep_all = TRUE)

# === Expand to ECG level ===
# amyloid: ALL ECGs (uncapped).  controls: 5 most-recent ECGs/patient.
amyloid_ecg <- data %>%
  filter(MRN %in% amyloid_cases_pat$MRN) %>%
  mutate(group = "amyloid")

control_ecg <- data %>%
  filter(MRN %in% controls_matched_pat$MRN) %>%
  arrange(MRN, desc(ECGDate)) %>%
  group_by(MRN) %>%
  slice_head(n = CONTROL_MAX_ECG) %>%
  ungroup() %>%
  mutate(group = "control")

final_matched <- bind_rows(amyloid_ecg, control_ecg)

# === Save ===
write_csv(final_matched, OUT_PATH)
cat("Final matched cohort saved to", OUT_PATH, "\n\n")

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
