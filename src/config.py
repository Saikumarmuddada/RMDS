"""Configuration and schema definitions for the RMDS Hackathon datasets."""

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT_DIR / "DOCs"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-specific column names and cost parameters."""

    name: str
    train_path: Path
    test_path: Path
    example_submission_path: Path
    id_col: str
    target_col: str
    disease_prefix: str
    vaccine_cost: float
    hospitalisation_cost: float


DATASET_C_CONFIG = DatasetConfig(
    name="Dataset C (COVID-19)",
    train_path=DOCS_DIR / "dataset_C_training.csv",
    test_path=DOCS_DIR / "dataset_C_testing.csv",
    example_submission_path=DOCS_DIR / "dataset_C_example_submission.csv",
    id_col="respondent_id",
    target_col="covid_vaccine",
    disease_prefix="covid",
    vaccine_cost=1500.0,
    hospitalisation_cost=41000.0,
)

DATASET_B_CONFIG = DatasetConfig(
    name="Dataset B (H1N1 Flu)",
    train_path=DOCS_DIR / "dataset_B_training.csv",
    test_path=DOCS_DIR / "dataset_B_testing.csv",
    example_submission_path=DOCS_DIR / "dataset_B_example_submission.csv",
    id_col="respondent_id",
    target_col="h1n1_vaccine",
    disease_prefix="h1n1",
    vaccine_cost=500.0,
    hospitalisation_cost=25000.0,
)


BEHAVIORAL_FEATURES = [
    "behavioral_antiviral_meds",
    "behavioral_avoidance",
    "behavioral_face_mask",
    "behavioral_wash_hands",
    "behavioral_large_gatherings",
    "behavioral_outside_home",
    "behavioral_touch_face",
]

OTHER_BINARY_FEATURES = [
    "child_under_6_months",
    "health_worker",
    "health_insurance",
]

CATEGORICAL_FEATURES = [
    "age_group",
    "education",
    "race",
    "sex",
    "income_poverty",
    "marital_status",
    "rent_or_own",
    "employment_status",
    "census_msa",
    "employment_sector",
]

AGE_ORDINAL_MAP = {
    "18 - 34 Years": 0.0,
    "35 - 44 Years": 1.0,
    "45 - 54 Years": 2.0,
    "55 - 64 Years": 3.0,
    "65+ Years": 4.0,
}

EDUCATION_ORDINAL_MAP = {
    "< 12 Years": 0.0,
    "12 Years": 1.0,
    "Some College": 2.0,
    "College Graduate": 3.0,
}

INCOME_ORDINAL_MAP = {
    "Below Poverty": 0.0,
    "<= $75,000, Above Poverty": 1.0,
    "> $75,000": 2.0,
}

RANDOM_SEED = 42
N_FOLDS = 5

