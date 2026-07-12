"""
Configuration and environment loading for IndustrialMind.

GEMINI_API_KEY must be set in the environment or in a `.env` file at the
project root (see .env.example). Never commit a real .env file.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    gemini_api_key: str
    extraction_model: str = "gemini-2.5-flash"
    explanation_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-004"


@lru_cache(maxsize=1)
def get_config() -> Config:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Create a .env file in the project "
            "root (see .env.example) with GEMINI_API_KEY=<your-key>, or "
            "export it as an environment variable."
        )
    return Config(
        gemini_api_key=api_key,
        extraction_model=os.environ.get("INDUSTRIALMIND_EXTRACTION_MODEL", "gemini-2.5-flash"),
        explanation_model=os.environ.get("INDUSTRIALMIND_EXPLANATION_MODEL", "gemini-2.5-flash"),
        embedding_model=os.environ.get("INDUSTRIALMIND_EMBEDDING_MODEL", "text-embedding-004"),
    )
