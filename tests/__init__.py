from pathlib import Path

TMP_DIR = Path(__file__).parent / '.tmp'
MODELS_DIR = TMP_DIR / 'models'
DATASETS_DIR = TMP_DIR / 'datasets'

TMP_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
