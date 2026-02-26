#!/usr/bin/env python3
"""
Template trainer for llama.cpp function calling.

Usage:
    python run.py --model /models/model.gguf --template glm
    python run.py --model /models/model.gguf --template /templates/custom.jinja --optimize
    python run.py --model /models/model.gguf --fetch-template Qwen/Qwen2.5-7B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from template_trainer.fetcher import get_template_for_model, fetch_template_from_repo, load_builtin_template
from template_trainer.llama_server import LlamaServer
from template_trainer.scorer import parse_tool_call, score_result, score_template, ToolCallResult
from template_trainer.optimizer import generate_variants, save_template, TemplateVariant
from tests.test_tools import TEST_CASES, get_test_cases_by_category


console = Console()


def run_tests(server: LlamaServer, test_cases: list) -> list[ToolCallResult]:
    """Run all test cases against the server."""
    results = []

    for tc in test_cases:
        try:
            response = server.chat(
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


def display_results(results: list[ToolCallResult], template_score):
    """Display test results in a formatted table."""
    table = Table(title=f"Template Score: {template_score.score:.1f}%")
    table.add_column("Test", style="cyan")
    table.add_column("Expected Tool", style="blue")
    table.add_column("Got Tool", style="green")
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
    console.print(f"  Passed: {template_score.passed}")
    console.print(f"  Partial: {template_score.partial}")
    console.print(f"  Failed: {template_score.failed}")
    console.print(f"  Score: {template_score.score:.1f}%")


def optimize_template(
    model_path: str,
    base_template: str,
    output_dir: Path,
    max_generations: int = 3,
    top_k: int = 3
) -> str:
    """Iteratively optimize template through mutations."""
    console.print("\n[bold cyan]Starting Template Optimization[/bold cyan]\n")

    best_variants = [TemplateVariant(
        template=base_template,
        score=0.0,
        generation=0
    )]

    for gen in range(max_generations):
        console.print(f"[bold]Generation {gen + 1}/{max_generations}[/bold]")

        # Generate new variants from top performers
        new_variants = []
        for variant in best_variants[:top_k]:
            new_variants.extend(generate_variants(
                variant.template,
                generation=gen,
                parent_id=f"gen{gen}_{variant.score:.0f}"
            ))

        # Test all variants
        all_to_test = best_variants[:top_k] + new_variants

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for i, variant in enumerate(all_to_test):
                task = progress.add_task(f"Testing variant {i + 1}/{len(all_to_test)}...", total=None)

                # Save template temporarily
                temp_path = Path(f"/tmp/template_gen{gen}_{i}.jinja")
                temp_path.write_text(variant.template)

                try:
                    with LlamaServer(
                        model_path=model_path,
                        template_path=str(temp_path),
                        port=8080 + gen * 100 + i
                    ) as server:
                        results = run_tests(server, TEST_CASES)
                        score = score_template(results)
                        variant.score = score.score
                except Exception as e:
                    console.print(f"[red]Error testing variant: {e}[/red]")
                    variant.score = 0.0

        # Sort and keep top performers
        all_to_test.sort(key=lambda v: v.score, reverse=True)
        best_variants = all_to_test[:top_k]

        console.print(f"  Best score: {best_variants[0].score:.1f}%")

        # Save best of this generation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = output_dir / f"template_gen{gen + 1}_{best_variants[0].score:.0f}.jinja"
        save_template(
            best_variants[0].template,
            save_path,
            metadata={
                "generation": gen + 1,
                "score": best_variants[0].score,
                "mutations": best_variants[0].mutations
            }
        )

    return best_variants[0].template


def main():
    parser = argparse.ArgumentParser(description="Train chat templates for tool calling")
    parser.add_argument("--model", required=True, help="Path to the GGUF model file")
    parser.add_argument("--template", help="Template name or path (e.g., 'glm' or '/path/to/template.jinja')")
    parser.add_argument("--fetch-template", help="Fetch template from HuggingFace repo (e.g., 'Qwen/Qwen2.5-7B-Instruct')")
    parser.add_argument("--optimize", action="store_true", help="Run optimization to improve template")
    parser.add_argument("--output-dir", default="/results", help="Directory to save results")
    parser.add_argument("--categories", help="Comma-separated test categories to run (basic,multi_tool,complex)")
    parser.add_argument("--gpu-layers", type=int, default=0, help="GPU layers for llama.cpp")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size for llama.cpp")

    args = parser.parse_args()

    # Get template
    if args.fetch_template:
        template, source = fetch_template_from_repo(args.fetch_template), f"fetched from {args.fetch_template}"
        if not template:
            console.print(f"[red]Could not fetch template from {args.fetch_template}[/red]")
            sys.exit(1)
    elif args.template:
        if Path(args.template).exists():
            template = Path(args.template).read_text()
            source = f"loaded from {args.template}"
        else:
            template = load_builtin_template(args.template)
            source = f"built-in {args.template} template"
    else:
        template, source = get_template_for_model(Path(args.model).stem)

    console.print(f"[cyan]Using template: {source}[/cyan]")

    # Get test cases
    if args.categories:
        categories = args.categories.split(",")
        test_cases = []
        for cat in categories:
            test_cases.extend(get_test_cases_by_category(cat.strip()))
    else:
        test_cases = TEST_CASES

    console.print(f"[cyan]Running {len(test_cases)} test cases[/cyan]\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.optimize:
        # Run optimization
        best_template = optimize_template(
            args.model,
            template,
            output_dir,
            max_generations=3
        )

        # Final test with best template
        console.print("\n[bold]Final evaluation with optimized template:[/bold]\n")
        template = best_template

    # Run tests
    temp_template = Path("/tmp/current_template.jinja")
    temp_template.write_text(template)

    with LlamaServer(
        model_path=args.model,
        template_path=str(temp_template),
        gpu_layers=args.gpu_layers,
        ctx_size=args.ctx_size
    ) as server:
        console.print("[green]Server started, running tests...[/green]\n")
        results = run_tests(server, test_cases)
        score = score_template(results)

    # Display results
    display_results(results, score)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"results_{timestamp}.json"
    result_data = {
        "template_source": source,
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
                "error": r.error
            }
            for r in results
        ]
    }
    result_file.write_text(json.dumps(result_data, indent=2))
    console.print(f"\n[green]Results saved to {result_file}[/green]")

    # Save template
    template_file = output_dir / f"template_{timestamp}.jinja"
    template_file.write_text(template)
    console.print(f"[green]Template saved to {template_file}[/green]")


if __name__ == "__main__":
    main()
