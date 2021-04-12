from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
REPORTS_DIR = PROJECT_DIR / 'reports'
REPORTS_FIGURES_DIR = REPORTS_DIR / 'figures'
MODELS_DIR = PROJECT_DIR / 'models'
