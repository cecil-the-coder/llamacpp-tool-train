"""Template optimization through iterative improvement."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import difflib


@dataclass
class TemplateVariant:
    """A template variant with its score."""
    template: str
    score: float
    generation: int
    parent_id: Optional[str] = None
    mutations: list[str] = None

    def __post_init__(self):
        if self.mutations is None:
            self.mutations = []


# Common improvements that can be applied to templates
TEMPLATE_MUTATIONS = {
    "add_tool_instructions": """You have access to tools. When you need to use a tool, output it in this exact format:
<tool_call={"name": "tool_name", "arguments": {"arg": "value"}}>""",

    "add_json_hint": "Always output valid JSON for tool arguments.",

    "add_strict_format": """IMPORTANT: Tool calls must be in this exact format:
<tool_call={"name": "function_name", "arguments": {"key": "value"}}>
Do not include any other text when making a tool call.""",

    "add_examples": """Examples:
User: What's the weather in Tokyo?
Assistant: <tool_call={"name": "get_weather", "arguments": {"location": "Tokyo"}}>""",

    "force_tool_first": "When a tool is available that could help, always use it rather than answering directly.",
}


def extract_system_prompt(template: str) -> tuple[str, str]:
    """Extract system prompt from template if present."""
    # Look for system message handling
    system_match = re.search(
        r'{%\s*if\s+message\[["\']role["\']\]\s*==\s*["\']system["\']s*%}(.*?){%\s*endif\s*%}',
        template,
        re.DOTALL
    )

    if system_match:
        return system_match.group(1).strip(), template

    return "", template


def add_system_instructions(template: str, instructions: str) -> str:
    """Add instructions to the system message handling in template."""
    # Find where to inject instructions
    # Look for the beginning of message handling

    if "{% for message in messages %}" in template:
        # Add before the loop
        injection = f"{{% set system_message = '{instructions}' %}}\n"
        return injection + template

    # Fallback: prepend to template
    return f"{{# System Instructions: {instructions} #}}\n{template}"


def mutate_template(template: str, mutation_name: str) -> str:
    """Apply a mutation to a template."""
    if mutation_name not in TEMPLATE_MUTATIONS:
        return template

    mutation = TEMPLATE_MUTATIONS[mutation_name]

    # Different mutation strategies
    if mutation_name == "add_tool_instructions":
        return add_system_instructions(template, mutation)
    elif mutation_name == "add_strict_format":
        return add_system_instructions(template, mutation)
    elif mutation_name == "add_examples":
        return add_system_instructions(template, mutation)
    elif mutation_name == "add_json_hint":
        return add_system_instructions(template, mutation)
    elif mutation_name == "force_tool_first":
        return add_system_instructions(template, mutation)

    return template


def generate_variants(
    base_template: str,
    generation: int = 0,
    parent_id: Optional[str] = None,
    mutations_to_try: Optional[list[str]] = None
) -> list[TemplateVariant]:
    """Generate template variants by applying mutations."""
    if mutations_to_try is None:
        mutations_to_try = list(TEMPLATE_MUTATIONS.keys())

    variants = []

    for mutation in mutations_to_try:
        mutated = mutate_template(base_template, mutation)
        if mutated != base_template:
            variants.append(TemplateVariant(
                template=mutated,
                score=0.0,
                generation=generation + 1,
                parent_id=parent_id,
                mutations=[mutation]
            ))

    return variants


def combine_mutations(base_template: str, mutations: list[str]) -> str:
    """Apply multiple mutations in sequence."""
    result = base_template
    for mutation in mutations:
        result = mutate_template(result, mutation)
    return result


def template_diff(template1: str, template2: str) -> str:
    """Generate a unified diff between two templates."""
    diff = difflib.unified_diff(
        template1.splitlines(keepends=True),
        template2.splitlines(keepends=True),
        fromfile="original",
        tofile="modified"
    )
    return "".join(diff)


def save_template(template: str, path: Path, metadata: Optional[dict] = None):
    """Save a template with optional metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if metadata:
        # Save as JSON with metadata
        data = {
            "template": template,
            **metadata
        }
        path.with_suffix(".json").write_text(json.dumps(data, indent=2))
    else:
        path.write_text(template)


def load_template(path: Path) -> tuple[str, Optional[dict]]:
    """Load a template, optionally with metadata."""
    json_path = path.with_suffix(".json")

    if json_path.exists():
        data = json.loads(json_path.read_text())
        template = data.pop("template")
        return template, data
    elif path.exists():
        return path.read_text(), None

    raise FileNotFoundError(f"Template not found: {path}")
