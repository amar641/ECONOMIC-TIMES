"""
IndustrialMind — AI for Industrial Knowledge Intelligence
============================================================
Unified Asset & Operations Brain built on the HMNN Evidence Engine.

Pipeline: Documents -> Extraction -> Classification -> {Knowledge Graph, HMNN}
          -> Industrial Brain (Asset Memory) -> RAG + Gemini (explanation layer)
"""

from industrialmind.config import get_config

__all__ = ["get_config"]
