from pathlib import Path

# The repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Standardized subdirectories
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"
MODELS_DIR = PROJECT_ROOT / "Models"
DATA_DIR = PROJECT_ROOT / "Supporting Data"
BACKEND_DIR = PROJECT_ROOT / "Application" / "Backend"
