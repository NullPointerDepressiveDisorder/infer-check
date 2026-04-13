import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console

from infer_check.types import Prompt

__all__ = ["load_suite", "save_suite"]

console = Console()


def load_suite(path: str | Path, num_prompts: int | None = None) -> list[Prompt]:
    """
    Read a JSONL file and validate each line against the Prompt model.
    Logs the count and category distribution via rich.console.
    If num_prompts is provided, selects an approximately equal number
    of prompts from each category.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Prompt suite not found: {path_obj}")

    all_prompts: list[Prompt] = []
    prompts_by_category: dict[str, list[Prompt]] = {}

    with path_obj.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                prompt = Prompt.model_validate(data)
                all_prompts.append(prompt)
                cat = prompt.category or "default"
                if cat not in prompts_by_category:
                    prompts_by_category[cat] = []
                prompts_by_category[cat].append(prompt)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path_obj}:{idx} - {e}") from e
            except ValidationError as e:
                raise ValueError(f"Invalid Prompt at {path_obj}:{idx} - {e}") from e

    # Apply num_prompts limit with equal category distribution
    if num_prompts is not None and num_prompts < len(all_prompts):
        selected_prompts: list[Prompt] = []
        categories = sorted(prompts_by_category.keys())
        num_categories = len(categories)

        if num_categories > 0:
            # Simple round-robin selection to keep categories equal
            # We iterate through categories and pick one prompt from each until we hit the limit
            # This ensures that even if categories have different sizes, we pick as equally as possible
            cat_indices = {cat: 0 for cat in categories}
            while len(selected_prompts) < num_prompts:
                added_in_round = False
                for cat in categories:
                    if len(selected_prompts) >= num_prompts:
                        break
                    idx = cat_indices[cat]
                    if idx < len(prompts_by_category[cat]):
                        selected_prompts.append(prompts_by_category[cat][idx])
                        cat_indices[cat] += 1
                        added_in_round = True
                if not added_in_round:
                    break
            final_prompts = selected_prompts
        else:
            final_prompts = all_prompts[:num_prompts]
    else:
        final_prompts = all_prompts

    # Log summary
    category_counts = Counter(p.category or "default" for p in final_prompts)
    console.print(f"[bold green]Loaded {len(final_prompts)} prompts from {path_obj.name}[/bold green]")
    for category, count in category_counts.most_common():
        console.print(f"  - {category}: {count}")

    return final_prompts


def save_suite(prompts: list[Prompt], path: str | Path) -> None:
    """Write a list of Prompts to a JSONL file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with path_obj.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            # exclude_none is useful if we don't want to save empty optional fields
            f.write(prompt.model_dump_json() + "\n")
