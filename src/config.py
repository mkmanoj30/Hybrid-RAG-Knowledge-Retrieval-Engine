# src/config.py

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Production Hybrid RAG Engine"
    VERSION: str = "0.1.0"

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"

    # Model & Retrieval Settings
    # all-MiniLM-L6-v2 is ultra-lightweight (384-d, ~80MB footprint)
    DENSE_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEVICE: str = "cpu"  # Fallback safe; can set to "cuda" if torch-cuda is available

    # Chunking Constraints
    TARGET_CHUNK_SIZE: int = 400      # Target token count per chunk
    CHUNK_OVERLAP: int = 50           # Overlap tokens between adjacent chunks
    MAX_CHUNK_SIZE: int = 600         # Upper hard limit for oversized sections

    # Reciprocal Rank Fusion (RRF) Constant
    # Standard constant k=60 prevents low-rank outliers from skewing balance
    RRF_K: int = 60
    TOP_K_RETRIEVAL: int = 5

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()

# Ensure directories exist upon runtime import
settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)