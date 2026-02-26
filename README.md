# llamacpp-tool-train

Test tool calling capabilities of llama.cpp models.

## Usage

```bash
# Build
docker build -t llamacpp-tool-train .

# Run against an existing llama.cpp server
docker run --rm llamacpp-tool-train --url http://glm-4-7-flash:8080

# Kubernetes
kubectl apply -f job.yaml
kubectl logs -f job/test-tools-glm-4-7-flash
```

## Test Categories

| Category | Description |
|----------|-------------|
| `basic` | Simple single tool calls |
| `multi_tool` | Multiple tools available, model must choose |
| `complex` | Complex arguments and expressions |
| `context` | Context-aware tool calls |
| `ambiguous` | Edge cases |

## Architecture

```
├── template_trainer/
│   ├── llama_server.py   # Client for llama.cpp API
│   ├── scorer.py         # Score tool call results
│   └── synbad_compat.py  # Synbad-compatible test cases
├── tests/
│   └── test_tools.py     # Test cases for tool calling
└── run.py                # Main entry point
```
