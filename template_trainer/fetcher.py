"""Fetch chat templates from HuggingFace model repositories."""

import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


# Known template mappings - which template format a model family uses
TEMPLATE_MAPPINGS = {
    "llama-3": "llama-3",
    "llama-3.1": "llama-3.1",
    "llama-3.2": "llama-3.1",
    "llama-3.3": "llama-3.1",
    "qwen2.5": "qwen2.5",
    "qwen2": "qwen2.5",
    "hermes": "hermes",
    "mistral": "mistral-v7",
    "deepseek-r1": "deepseek-r1",
    "glm": "glm",
    "phi": "phi-4",
    "gemma": "gemma",
}


def fetch_tokenizer_config(repo_id: str) -> dict:
    """Fetch tokenizer_config.json from a HuggingFace repo."""
    url = f"https://huggingface.co/{repo_id}/raw/main/tokenizer_config.json"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError:
        return {}


def fetch_template_from_repo(repo_id: str) -> Optional[str]:
    """Extract chat template from a HuggingFace model repo."""
    config = fetch_tokenizer_config(repo_id)

    # Check for chat_template in tokenizer_config
    if "chat_template" in config:
        template = config["chat_template"]
        # Could be a string or list of templates
        if isinstance(template, str):
            return template
        elif isinstance(template, list):
            # Find default or first template
            for t in template:
                if isinstance(t, dict) and t.get("name") == "default":
                    return t.get("template")
            if template:
                return template[0].get("template") if isinstance(template[0], dict) else template[0]

    return None


def guess_template_family(model_name: str) -> str:
    """Guess which template family a model belongs to based on name."""
    model_lower = model_name.lower()

    for key, family in TEMPLATE_MAPPINGS.items():
        if key in model_lower:
            return family

    return "generic"


def load_builtin_template(template_name: str) -> str:
    """Load a built-in template from the templates directory."""
    template_path = Path(__file__).parent.parent / "templates" / f"{template_name}.jinja"
    if template_path.exists():
        return template_path.read_text()

    # Fall back to generic
    generic_path = Path(__file__).parent.parent / "templates" / "generic.jinja"
    if generic_path.exists():
        return generic_path.read_text()

    # Return minimal fallback
    return """{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}{% endfor %}"""


def get_template_for_model(model_name: str, repo_id: Optional[str] = None) -> tuple[str, str]:
    """
    Get the best template for a model.

    Returns: (template_content, source_description)
    """
    # Try to fetch from repo if provided
    if repo_id:
        template = fetch_template_from_repo(repo_id)
        if template:
            return template, f"fetched from {repo_id}"

    # Guess family and load built-in
    family = guess_template_family(model_name)
    template = load_builtin_template(family)

    return template, f"built-in {family} template"
