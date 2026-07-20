"""
Configuration and environment loading for IndustrialMind.

IndustrialMind uses a locally running Ollama server. Configure its URL and
model in `.env` if the defaults do not match your Docker setup.
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_timeout_seconds: int = 180


@lru_cache(maxsize=1)
def get_config() -> Config:
    try:
        timeout = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "180"))
    except ValueError as exc:
        raise RuntimeError("OLLAMA_TIMEOUT_SECONDS must be a whole number.") from exc
    return Config(
        ollama_base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:7b"),
        ollama_timeout_seconds=timeout,
    )
