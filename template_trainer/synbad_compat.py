"""
Synbad-compatible test cases for tool calling.

These test cases are compatible with the synbad evaluation format
and cover various tool calling scenarios.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SynbadTestCase:
    """Synbad-compatible test case structure."""
    name: str
    description: str
    json_data: dict  # The request JSON (messages, tools, parameters)
    expected_tool_calls: list[dict]  # Expected tool calls
    category: str = "tools"


# Convert from our test format to synbad-compatible format
def to_synbad_format(test_case) -> dict:
    """Convert internal test case to synbad JSON format."""
    return {
        "messages": test_case.messages,
        "tools": test_case.tools,
        "tool_choice": "auto",
        "temperature": 0.0,
    }


# Synbad-compatible test cases (matching their patterns)

SYNBAD_COMPAT_TESTS = [
    # Simple tool call (like synbad's simple-tool.ts)
    SynbadTestCase(
        name="simple-tool",
        description="Basic single tool call",
        json_data={
            "messages": [
                {"role": "user", "content": "What's the weather like in Paris?"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city, e.g. Paris"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto",
            "temperature": 0.0,
        },
        expected_tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Paris"}}
        ],
        category="tools"
    ),

    # Parallel tool calls (like synbad's parallel-tool.ts)
    SynbadTestCase(
        name="parallel-tool",
        description="Multiple parallel tool calls",
        json_data={
            "messages": [
                {"role": "user", "content": "What's the weather like in Paris and London?"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "temperature": 0.0,
        },
        expected_tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Paris"}},
            {"name": "get_weather", "arguments": {"location": "London"}},
        ],
        category="tools"
    ),

    # Tool with dashes in name (like synbad's tool-dash-underscore.ts)
    SynbadTestCase(
        name="tool-dash-underscore",
        description="Tool name with dashes and underscores",
        json_data={
            "messages": [
                {"role": "user", "content": "Get the weather using the v1 API"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get-weather__v1",
                    "description": "Get weather using v1 API",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto",
            "temperature": 0.0,
        },
        expected_tool_calls=[
            {"name": "get-weather__v1", "arguments": {"location": None}}  # location required but not specified by user
        ],
        category="tools"
    ),

    # Tool with no arguments (like synbad's no-fn-args.ts)
    SynbadTestCase(
        name="no-fn-args",
        description="Tool with no required arguments",
        json_data={
            "messages": [
                {"role": "user", "content": "Get the current time"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }],
            "tool_choice": "auto",
            "temperature": 0.0,
        },
        expected_tool_calls=[
            {"name": "get_current_time", "arguments": {}}
        ],
        category="tools"
    ),

    # Multi-turn tool conversation (like synbad's multi-turn-tools.ts)
    SynbadTestCase(
        name="multi-turn-tools",
        description="Multi-turn conversation with tools",
        json_data={
            "messages": [
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo"}'
                    }
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 22°C"},
                {"role": "user", "content": "What about Kyoto?"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        },
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto",
            "temperature": 0.0,
        },
        expected_tool_calls=[
            {"name": "get_weather", "arguments": {"location": "Kyoto"}}
        ],
        category="tools"
    ),
]


def get_synbad_tests() -> list[SynbadTestCase]:
    """Get all synbad-compatible test cases."""
    return SYNBAD_COMPAT_TESTS
