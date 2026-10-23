"""Central configuration: paths and constants used across the project."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=value lines) so no extra dependency is needed."""
    if not path.is_file():  # skip if missing, or if ".env" is a virtual-environment folder
        return
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.split(" #", 1)[0].strip().strip('"').strip("'")  # drop inline comments
            os.environ.setdefault(k.strip(), v)


_load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DB_PATH = PROCESSED_DIR / "housing.duckdb"
RAW_CSV = RAW_DIR / "PPR-ALL.csv"

PRICE_MODEL_PATH = MODELS_DIR / "price_model.joblib"
PRICE_METRICS_PATH = MODELS_DIR / "price_metrics.json"
FORECAST_PATH = MODELS_DIR / "forecasts.csv"
FORECAST_METRICS_PATH = MODELS_DIR / "forecast_metrics.json"
LLM_EVAL_PATH = MODELS_DIR / "llm_eval.json"

# Official Property Price Register bulk download (all sales since 2010)
PPR_URL = (
    "https://www.propertypriceregister.ie/website/npsra/ppr/npsra-ppr.nsf/"
    "Downloads/PPR-ALL.zip/$FILE/PPR-ALL.zip"
)

COUNTIES = [
    "Carlow", "Cavan", "Clare", "Cork", "Donegal", "Dublin", "Galway", "Kerry",
    "Kildare", "Kilkenny", "Laois", "Leitrim", "Limerick", "Longford", "Louth",
    "Mayo", "Meath", "Monaghan", "Offaly", "Roscommon", "Sligo", "Tipperary",
    "Waterford", "Westmeath", "Wexford", "Wicklow",
]

# New builds in the PPR are often reported excluding VAT (13.5% on residential)
VAT_RATE = 0.135

# Remove obviously non-residential / data-entry outliers
MIN_PRICE = 30_000
MAX_PRICE = 5_000_000

# LLM settings (override in .env)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai | ollama
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
