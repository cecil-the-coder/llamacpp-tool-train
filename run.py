#!/usr/bin/env python3
"""
Template tester for llama.cpp function calling.

Usage:
    python run.py --url http://glm-4-7-flash.inference.svc.cluster.local:8080
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table

from template_trainer.llama_server import LlamaClient
from template_trainer.scorer import parse_tool_call, score_result, score_template, ToolCallResult
from tests.test_tools import TEST_CASES, get_test_cases_by_category


console = Console()


def run_tests(client: LlamaClient, test_cases: list) -> list[ToolCallResult]:
    """Run all test cases against the server."""
    results = []

    for tc in test_cases:
        try:
            response = client.chat(
                messages=tc.messages,
                tools=tc.tools,
                temperature=0.0,
                max_tokens=256
            )

            # Extract the assistant's response
            message = response.get("choices", [{}])[0].get("message", {})
            content = message.get("content", "")

            # Check for tool_calls in response (native format)
            if "tool_calls" in message and message["tool_calls"]:
                tool_call = message["tool_calls"][0]
                tool_name = tool_call.get("function", {}).get("name")
                args_str = tool_call.get("function", {}).get("arguments", "{}")
                try:
                    arguments = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    arguments = {"raw": args_str}

                result = ToolCallResult(
                    test_name=tc.name,
                    success=True,
                    tool_name=tool_name,
                    arguments=arguments,
                    raw_output=content
                )
            else:
                # Parse from content
                tool_name, arguments, _ = parse_tool_call(content)
                result = ToolCallResult(
                    test_name=tc.name,
                    success=bool(tool_name),
                    tool_name=tool_name,
                    arguments=arguments,
                    raw_output=content
                )

            # Score the result
            result = score_result(result, tc.expected_tool, tc.expected_args)
            results.append(result)

        except Exception as e:
            results.append(ToolCallResult(
                test_name=tc.name,
                success=False,
                error=str(e)
            ))

    return results


def display_results(results: list[ToolCallResult], score):
    """Display test results in a formatted table."""
    table = Table(title=f"Tool Calling Score: {score.score:.1f}%")
    table.add_column("Test", style="cyan")
    table.add_column("Expected", style="blue")
    table.add_column("Got", style="green")
    table.add_column("Score", style="yellow")
    table.add_column("Status", style="bold")

    for r in results:
        tc = next((t for t in TEST_CASES if t.name == r.test_name), None)
        expected = tc.expected_tool if tc else "?"

        status = "✓" if r.success else "✗"
        status_style = "green" if r.success else "red"

        table.add_row(
            r.test_name,
            expected,
            r.tool_name or "(none)",
            f"{r.score:.0f}",
            f"[{status_style}]{status}[/{status_style}]"
        )

    console.print(table)

    # Print summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Passed: {score.passed}")
    console.print(f"  Partial: {score.partial}")
    console.print(f"  Failed: {score.failed}")
    console.print(f"  Score: {score.score:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Test tool calling on llama.cpp server")
    parser.add_argument("--url", required=True, help="URL of the llama.cpp server (e.g., http://glm-4-7-flash:8080)")
    parser.add_argument("--output-dir", default="/results", help="Directory to save results")
    parser.add_argument("--categories", help="Comma-separated test categories to run (basic,multi_tool,complex)")

    args = parser.parse_args()

    # Connect to server
    client = LlamaClient(args.url)

    console.print(f"[cyan]Connecting to {args.url}...[/cyan]")
    if not client.health_check():
        console.print(f"[red]Cannot connect to server[/red]")
        sys.exit(1)
    console.print("[green]Connected![/green]\n")

    # Get test cases
    if args.categories:
        categories = args.categories.split(",")
        test_cases = []
        for cat in categories:
            test_cases.extend(get_test_cases_by_category(cat.strip()))
    else:
        test_cases = TEST_CASES

    console.print(f"[cyan]Running {len(test_cases)} test cases[/cyan]\n")

    # Run tests
    results = run_tests(client, test_cases)
    score = score_template(results)

    # Display results
    display_results(results, score)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"results_{timestamp}.json"
    result_data = {
        "url": args.url,
        "timestamp": timestamp,
        "score": score.score,
        "passed": score.passed,
        "partial": score.partial,
        "failed": score.failed,
        "results": [
            {
                "test": r.test_name,
                "success": r.success,
                "tool_name": r.tool_name,
                "arguments": r.arguments,
                "score": r.score,
                "raw_output": r.raw_output,
                "error": r.error
            }
            for r in results
        ]
    }
    result_file.write_text(json.dumps(result_data, indent=2))
    console.print(f"\n[green]Results saved to {result_file}[/green]")


if __name__ == "__main__":
    main()
