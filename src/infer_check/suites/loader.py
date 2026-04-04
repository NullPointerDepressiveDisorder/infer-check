import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console

from infer_check.types import Prompt

__all__ = ["load_suite", "save_suite"]

console = Console()


def load_suite(path: str | Path) -> list[Prompt]:
    """
    Read a JSONL file and validate each line against the Prompt model.
    Logs the count and category distribution via rich.console.
    Raises ValueError with the line number on invalid entries.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Prompt suite not found: {path_obj}")

    prompts = []
    category_counts: Counter[str] = Counter()

    with path_obj.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                prompt = Prompt.model_validate(data)
                # Keep track of which fields were explicitly set in the input JSONL
                prompt.metadata["__fields_set__"] = set(data.keys())
                prompts.append(prompt)
                category_counts[prompt.category] += 1
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path_obj}:{idx} - {e}") from e
            except ValidationError as e:
                raise ValueError(f"Invalid Prompt at {path_obj}:{idx} - {e}") from e

    # Log summary
    console.print(f"[bold green]Loaded {len(prompts)} prompts from {path_obj.name}[/bold green]")
    for category, count in category_counts.most_common():
        console.print(f"  - {category}: {count}")

    return prompts


def save_suite(prompts: list[Prompt], path: str | Path) -> None:
    """Write a list of Prompts to a JSONL file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            # exclude_none is useful if we don't want to save empty optional fields
            f.write(prompt.model_dump_json() + "\n")
