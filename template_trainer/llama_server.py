"""Manage llama.cpp server for template testing."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional
import httpx


class LlamaServer:
    """Context manager for llama.cpp server."""

    def __init__(
        self,
        model_path: str,
        template_path: Optional[str] = None,
        port: int = 8080,
        host: str = "127.0.0.1",
        gpu_layers: int = 0,
        ctx_size: int = 4096,
        llamacpp_dir: str = "/app/llama.cpp"
    ):
        self.model_path = model_path
        self.template_path = template_path
        self.port = port
        self.host = host
        self.gpu_layers = gpu_layers
        self.ctx_size = ctx_size
        self.llamacpp_dir = llamacpp_dir
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"

    def start(self, timeout: float = 60.0) -> bool:
        """Start the llama.cpp server."""
        cmd = [
            f"{self.llamacpp_dir}/llama-server",
            "--model", self.model_path,
            "--port", str(self.port),
            "--host", self.host,
            "--ctx-size", str(self.ctx_size),
            "--jinja",  # Enable Jinja for function calling
        ]

        if self.gpu_layers > 0:
            cmd.extend(["--n-gpu-layers", str(self.gpu_layers)])

        if self.template_path:
            cmd.extend(["--chat-template-file", self.template_path])

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.llamacpp_dir
        )

        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = httpx.get(f"{self.base_url}/health", timeout=2.0)
                if response.status_code == 200:
                    return True
            except (httpx.ConnectError, httpx.ReadTimeout):
                pass
            time.sleep(0.5)

        # Timeout - check if process is still running
        if self.process.poll() is not None:
            stderr = self.process.stderr.read().decode() if self.process.stderr else ""
            raise RuntimeError(f"Server failed to start: {stderr}")

        raise RuntimeError(f"Server did not become ready within {timeout}s")

    def stop(self):
        """Stop the llama.cpp server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

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
