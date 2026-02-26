"""Score template performance based on tool calling tests."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCallResult:
    """Result of a single tool call test."""
    test_name: str
    success: bool
    tool_name: Optional[str] = None
    arguments: Optional[dict] = None
    raw_output: str = ""
    error: str = ""
    score: float = 0.0


@dataclass
class TemplateScore:
    """Overall score for a template."""
    template_name: str
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    partial: int = 0
    results: list[ToolCallResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Calculate overall score (0-100)."""
        if self.total_tests == 0:
            return 0.0
        # Full credit for passed, half for partial
        points = self.passed + (self.partial * 0.5)
        return (points / self.total_tests) * 100


def parse_tool_call(output: str) -> tuple[Optional[str], Optional[dict], str]:
    """
    Parse tool call from model output.

    Returns: (tool_name, arguments, raw_match)
    """
    # Try various formats

    # Format 1: <tool_call={"name": "...", "arguments": {...}}>
    pattern1 = r'<tool_call=(\{[^}]+\})>'
    match1 = re.search(pattern1, output, re.DOTALL)
    if match1:
        try:
            data = json.loads(match1.group(1))
            return data.get("name"), data.get("arguments", {}), match1.group(0)
        except json.JSONDecodeError:
            pass

    # Format 2: {"name": "...", "arguments": {...}} (standalone JSON)
    pattern2 = r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
    match2 = re.search(pattern2, output, re.DOTALL)
    if match2:
        try:
            tool_name = match2.group(1)
            arguments = json.loads(match2.group(2))
            return tool_name, arguments, match2.group(0)
        except json.JSONDecodeError:
            pass

    # Format 3: ΤΟΟL_CALL: name(args) or similar
    pattern3 = r'(?:tool_call|function_call)[:\s]+(\w+)\s*\(([^)]*)\)'
    match3 = re.search(pattern3, output, re.IGNORECASE)
    if match3:
        tool_name = match3.group(1)
        args_str = match3.group(2)
        # Try to parse args as JSON or key=value
        try:
            if args_str.strip().startswith('{'):
                arguments = json.loads(args_str)
            else:
                arguments = {"raw": args_str}
            return tool_name, arguments, match3.group(0)
        except json.JSONDecodeError:
            return tool_name, {"raw": args_str}, match3.group(0)

    # Format 4: XML-style <function=name>args</function>
    pattern4 = r'<function=(\w+)>([^<]*)</function>'
    match4 = re.search(pattern4, output, re.DOTALL)
    if match4:
        tool_name = match4.group(1)
        args_str = match4.group(2).strip()
        try:
            arguments = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            arguments = {"raw": args_str}
        return tool_name, arguments, match4.group(0)

    return None, None, ""


def score_result(result: ToolCallResult, expected_tool: str, expected_args: dict) -> ToolCallResult:
    """Score a tool call result against expected values."""
    result.score = 0.0

    if not result.tool_name:
        result.error = "No tool call detected in output"
        return result

    # Score tool name match
    if result.tool_name == expected_tool:
        result.score += 50.0
    elif result.tool_name.lower() == expected_tool.lower():
        result.score += 40.0
    else:
        result.error = f"Wrong tool: expected {expected_tool}, got {result.tool_name}"
        return result

    # Score argument match
    if not expected_args:
        # No args expected
        result.score += 50.0
        result.success = True
        return result

    if not result.arguments:
        result.error = "Expected arguments but none provided"
        return result

    # Check each expected argument
    arg_score = 0
    total_args = len(expected_args)
    matched_args = 0

    for key, expected_value in expected_args.items():
        if key in result.arguments:
            actual_value = result.arguments[key]
            if actual_value == expected_value:
                matched_args += 1
            elif str(actual_value).lower() == str(expected_value).lower():
                matched_args += 0.8  # Partial match
            else:
                matched_args += 0.3  # Key exists but value different
        else:
            # Check for similar keys
            for actual_key in result.arguments:
                if key.lower() in actual_key.lower() or actual_key.lower() in key.lower():
                    matched_args += 0.2
                    break

    arg_score = (matched_args / total_args) * 50.0 if total_args > 0 else 50.0
    result.score += arg_score

    if result.score >= 90:
        result.success = True
    elif result.score >= 50:
        result.success = True  # Partial credit still counts

    return result


def score_template(results: list[ToolCallResult], template_name: str = "unknown") -> TemplateScore:
    """Calculate overall score for a template based on all test results."""
    score = TemplateScore(template_name=template_name)

    for result in results:
        score.total_tests += 1
        score.results.append(result)

        if result.success:
            if result.score >= 90:
                score.passed += 1
            else:
                score.partial += 1
        else:
            score.failed += 1

    return score
