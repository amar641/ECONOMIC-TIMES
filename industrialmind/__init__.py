"""
IndustrialMind — AI for Industrial Knowledge Intelligence
============================================================
Unified Asset & Operations Brain built on the HMNN Evidence Engine.

Pipeline: Documents -> Extraction -> Classification -> {Knowledge Graph, HMNN}
          -> Industrial Brain (Asset Memory) -> local RAG + Ollama (explanation layer)
"""

from industrialmind.config import get_config

__all__ = ["get_config"]
