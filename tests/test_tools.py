"""Test cases for tool calling functionality."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ToolTestCase:
    """A test case for tool calling."""
    name: str
    description: str
    messages: list[dict]
    tools: list[dict]
    expected_tool: str
    expected_args: dict
    category: str = "basic"


# Basic tool definitions for testing
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g., 'Tokyo, Japan'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    }
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return"
                }
            },
            "required": ["query"]
        }
    }
}


# Test cases organized by category
TEST_CASES = [
    # Basic single tool tests
    ToolTestCase(
        name="simple_weather",
        description="Simple weather query",
        messages=[
            {"role": "user", "content": "What's the weather like in Tokyo?"}
        ],
        tools=[WEATHER_TOOL],
        expected_tool="get_weather",
        expected_args={"location": "Tokyo"},
        category="basic"
    ),
    ToolTestCase(
        name="weather_with_unit",
        description="Weather query specifying unit",
        messages=[
            {"role": "user", "content": "What's the weather in Paris in fahrenheit?"}
        ],
        tools=[WEATHER_TOOL],
        expected_tool="get_weather",
        expected_args={"location": "Paris", "unit": "fahrenheit"},
        category="basic"
    ),
    ToolTestCase(
        name="simple_calculation",
        description="Simple math calculation",
        messages=[
            {"role": "user", "content": "What is 15 times 7?"}
        ],
        tools=[CALCULATOR_TOOL],
        expected_tool="calculate",
        expected_args={"expression": "15 * 7"},
        category="basic"
    ),
    ToolTestCase(
        name="web_search",
        description="Simple web search",
        messages=[
            {"role": "user", "content": "Search for the latest news about AI"}
        ],
        tools=[SEARCH_TOOL],
        expected_tool="web_search",
        expected_args={"query": "latest news about AI"},
        category="basic"
    ),

    # Multi-tool selection tests
    ToolTestCase(
        name="choose_weather",
        description="Choose weather tool from multiple options",
        messages=[
            {"role": "user", "content": "Is it raining in London?"}
        ],
        tools=[WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL],
        expected_tool="get_weather",
        expected_args={"location": "London"},
        category="multi_tool"
    ),
    ToolTestCase(
        name="choose_calculator",
        description="Choose calculator tool from multiple options",
        messages=[
            {"role": "user", "content": "Calculate 123 plus 456"}
        ],
        tools=[WEATHER_TOOL, CALCULATOR_TOOL, SEARCH_TOOL],
        expected_tool="calculate",
        expected_args={"expression": "123 + 456"},
        category="multi_tool"
    ),

    # Complex argument tests
    ToolTestCase(
        name="detailed_location",
        description="Weather with detailed location",
        messages=[
            {"role": "user", "content": "What's the weather in New York City, USA?"}
        ],
        tools=[WEATHER_TOOL],
        expected_tool="get_weather",
        expected_args={"location": "New York City, USA"},
        category="complex"
    ),
    ToolTestCase(
        name="complex_calculation",
        description="Complex mathematical expression",
        messages=[
            {"role": "user", "content": "Calculate the square root of 144 plus 25"}
        ],
        tools=[CALCULATOR_TOOL],
        expected_tool="calculate",
        expected_args={"expression": "sqrt(144) + 25"},
        category="complex"
    ),

    # Conversation context tests
    ToolTestCase(
        name="context_aware",
        description="Tool call with conversation context",
        messages=[
            {"role": "user", "content": "I'm planning a trip to Berlin."},
            {"role": "assistant", "content": "That sounds exciting! Berlin is a wonderful city with rich history."},
            {"role": "user", "content": "What's the weather like there?"}
        ],
        tools=[WEATHER_TOOL],
        expected_tool="get_weather",
        expected_args={"location": "Berlin"},
        category="context"
    ),

    # Ambiguous cases
    ToolTestCase(
        name="ambiguous_location",
        description="Handle ambiguous location",
        messages=[
            {"role": "user", "content": "What's the weather in Springfield?"}
        ],
        tools=[WEATHER_TOOL],
        expected_tool="get_weather",
        expected_args={"location": "Springfield"},  # Model should pick one
        category="ambiguous"
    ),
]


def get_test_cases_by_category(category: str) -> list[ToolTestCase]:
    """Filter test cases by category."""
    return [tc for tc in TEST_CASES if tc.category == category]


def get_all_test_cases() -> list[ToolTestCase]:
    """Get all test cases."""
    return TEST_CASES
