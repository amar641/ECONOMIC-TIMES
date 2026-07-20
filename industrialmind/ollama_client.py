"""Small dependency-free client for a local Ollama server."""

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from industrialmind.config import Config


class OllamaClient:
    def __init__(self, config: Config):
        self._base_url = config.ollama_base_url
        self._model = config.ollama_model
        self._timeout = config.ollama_timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    def ensure_ready(self) -> None:
        """Confirm that Ollama is reachable and the configured model is present."""
        try:
            with urlopen(f"{self._base_url}/api/tags", timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self._base_url}. Start the Ollama Docker "
                "container and expose port 11434."
            ) from exc

        installed = {item.get("name") for item in payload.get("models", [])}
        if self._model not in installed:
            raise RuntimeError(
                f"Ollama model '{self._model}' is not available at {self._base_url}. "
                "Run `ollama pull qwen2.5:7b` inside the Ollama environment, or set "
                "OLLAMA_MODEL to an installed model."
            )

    def generate(self, prompt: str, system: Optional[str] = None,
                 json_output: bool = False, temperature: float = 0.2) -> str:
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            body["system"] = system
        if json_output:
            body["format"] = "json"
        request = Request(
            f"{self._base_url}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama generation failed using model '{self._model}'.") from exc
        answer = result.get("response", "").strip()
        if not answer:
            raise RuntimeError(f"Ollama model '{self._model}' returned an empty response.")
        return answer
