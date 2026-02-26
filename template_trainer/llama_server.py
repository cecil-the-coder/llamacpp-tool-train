"""Client for llama.cpp / LiteLLM server."""

import httpx
from typing import Optional


class LlamaClient:
    """Client for connecting to an OpenAI-compatible server (llama.cpp or LiteLLM)."""

    def __init__(self, base_url: str, model: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def health_check(self, timeout: float = 10.0) -> bool:
        """Verify connection to server."""
        try:
            response = httpx.get(f"{self.base_url}/health", timeout=timeout)
            return response.status_code == 200
        except (httpx.ConnectError, httpx.ReadTimeout):
            # Try LiteLLM health endpoints
            try:
                response = httpx.get(f"{self.base_url}/health/liveliness", timeout=timeout)
                return response.status_code == 200
            except:
                raise RuntimeError(f"Cannot connect to server at {self.base_url}")

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        tool_choice: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> dict:
        """Send a chat completion request."""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if self.model:
            payload["model"] = self.model

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=120.0
        )
        response.raise_for_status()
        return response.json()
