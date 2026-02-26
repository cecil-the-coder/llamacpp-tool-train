# llamacpp-tool-train

Test tool calling capabilities of llama.cpp models via LiteLLM.

## Usage

```bash
# Port-forward to LiteLLM
kubectl port-forward -n inference svc/litellm 4000:4000 &

# Run tests
python3 run.py --url http://localhost:4000 --model glm-4-7-flash

# With verbose output
python3 run.py --url http://localhost:4000 --model glm-4-7-flash -v
```

## Test Categories

| Category | Description |
|----------|-------------|
| `basic` | Simple single tool calls |
| `multi_tool` | Multiple tools available, model must choose |
| `complex` | Complex arguments and expressions |
| `context` | Context-aware tool calls |
| `ambiguous` | Edge cases |

## Requirements

```bash
pip install httpx rich
```
