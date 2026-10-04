"""Thin wrapper over the official Ollama Python client (local loopback only)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from clawde.config import OllamaConfig
from clawde.errors import InterpreterError


@dataclass(frozen=True)
class OllamaProbe:
    reachable: bool
    models: tuple[str, ...]
    error: str | None = None

    def has_model(self, name: str) -> bool:
        return name in self.models or f"{name}:latest" in self.models


def _is_cloud(name: str) -> bool:
    tag = name.split(":")[-1]
    return "cloud" in tag or name.endswith("-cloud")


class OllamaBrain:
    """Structured chat against a locally installed model."""

    def __init__(self, config: OllamaConfig, client: Any | None = None) -> None:
        self.config = config
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import ollama  # deferred: not needed for --help or rule-based runs

            self._client = ollama.Client(host=self.config.host, timeout=self.config.timeout_s)
        return self._client

    def probe(self, timeout_s: float = 2.0) -> OllamaProbe:
        """Check the local server and list installed, non-cloud models."""
        try:
            if self._client is None:
                import ollama

                client = ollama.Client(host=self.config.host, timeout=timeout_s)
            else:
                client = self._client
            listing = client.list()
        except Exception as exc:  # connection refused, timeout, missing package
            return OllamaProbe(reachable=False, models=(), error=str(exc) or type(exc).__name__)
        names = []
        for model in getattr(listing, "models", []) or []:
            name = getattr(model, "model", None)
            if name and not _is_cloud(name):
                names.append(name)
        return OllamaProbe(reachable=True, models=tuple(sorted(names)))

    def chat_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        """Run one structured chat request. Returns the raw JSON text."""
        try:
            response = self.client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format=schema,
                think=self.config.think,
                options={"temperature": self.config.temperature},
            )
        except Exception as exc:
            raise InterpreterError(f"Ollama no respondió correctamente: {exc}") from exc
        content = response.message.content if hasattr(response, "message") else None
        if not content:
            raise InterpreterError("Ollama devolvió una respuesta vacía")
        return content
