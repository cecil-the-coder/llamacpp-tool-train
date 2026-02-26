# llamacpp-tool-train

A containerized system for systematically improving chat templates for function calling in llama.cpp.

## Overview

This tool helps you:

1. **Fetch templates** from HuggingFace model repositories or use built-in templates
2. **Test templates** against a comprehensive test suite for tool calling
3. **Optimize templates** through iterative mutation and scoring

## Quick Start

```bash
# Build the container
docker build -t llamacpp-tool-train .

# Run basic evaluation
docker run --rm \
  -v /path/to/models:/models \
  -v $(pwd)/results:/results \
  llamacpp-tool-train \
  --model /models/glm-4-7b.gguf \
  --template glm

# Fetch template from HuggingFace
docker run --rm \
  -v /path/to/models:/models \
  -v $(pwd)/results:/results \
  llamacpp-tool-train \
  --model /models/model.gguf \
  --fetch-template Qwen/Qwen2.5-7B-Instruct

# Run optimization to find better templates
docker run --rm \
  -v /path/to/models:/models \
  -v $(pwd)/results:/results \
  llamacpp-tool-train \
  --model /models/model.gguf \
  --template generic \
  --optimize
```

## Test Categories

| Category | Description |
|----------|-------------|
| `basic` | Simple single tool calls |
| `multi_tool` | Multiple tools available, model must choose |
| `complex` | Complex arguments and expressions |
| `context` | Context-aware tool calls |
| `ambiguous` | Edge cases and ambiguous inputs |

## Built-in Templates

- `generic` - Generic template for models without native tool support
- `llama-3.1` - Llama 3.1/3.2/3.3 native tool calling
- `qwen2.5` - Qwen 2.5 native tool calling
- `glm` - GLM-4 style template

## Architecture

```
├── template_trainer/
│   ├── fetcher.py      # Fetch templates from HuggingFace
│   ├── llama_server.py # Manage llama.cpp server
│   ├── scorer.py       # Score tool call results
│   ├── optimizer.py    # Template mutation/optimization
│   └── synbad_compat.py # Synbad-compatible test cases
├── templates/          # Built-in Jinja templates
├── tests/
│   └── test_tools.py   # Test cases for tool calling
└── run.py              # Main entry point
```

## Synbad Compatibility

This project includes test cases compatible with [synbad](https://github.com/synthetic-lab/synbad), an evaluation framework for detecting bugs in LLM inference providers. The test patterns cover:

- Simple tool calls
- Parallel tool calls
- Multi-turn tool conversations
- Edge cases (dashes in names, no arguments, etc.)

## How Optimization Works

1. Start with a base template
2. Apply mutations (add instructions, examples, strict formatting)
3. Test each variant against the test suite
4. Keep top performers and generate new variants
5. Repeat for N generations
6. Output the best-performing template

Mutations include:
- Adding tool use instructions
- Adding strict format requirements
- Adding examples
- Forcing tool-first behavior
