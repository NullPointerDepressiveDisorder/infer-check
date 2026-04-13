import json
from pathlib import Path

from infer_check.suites.loader import load_suite


def test_load_suite_equal_distribution(tmp_path: Path) -> None:
    """Test that load_suite distributes num_prompts equally across categories."""
    prompt_file = tmp_path / "test_prompts.jsonl"

    # 10 math, 5 code, 2 logic
    prompts = []
    for i in range(10):
        prompts.append({"id": f"math-{i}", "text": f"math {i}", "category": "math"})
    for i in range(5):
        prompts.append({"id": f"code-{i}", "text": f"code {i}", "category": "code"})
    for i in range(2):
        prompts.append({"id": f"logic-{i}", "text": f"logic {i}", "category": "logic"})

    prompt_file.write_text("\n".join(json.dumps(p) for p in prompts))

    # Request 6 prompts.
    # Round 1: math-0, code-0, logic-0 (3 total)
    # Round 2: math-1, code-1, logic-1 (6 total)
    # Categories: code, logic, math (sorted)
    # Round 1: code-0, logic-0, math-0
    # Round 2: code-1, logic-1, math-1
    loaded = load_suite(prompt_file, num_prompts=6)

    assert len(loaded) == 6
    categories = [p.category for p in loaded]
    from collections import Counter

    counts = Counter(categories)

    assert counts["math"] == 2
    assert counts["code"] == 2
    assert counts["logic"] == 2

    # Request 4 prompts
    # Round 1: code-0, logic-0, math-0 (3 total)
    # Round 2: code-1 (4 total)
    loaded_4 = load_suite(prompt_file, num_prompts=4)
    assert len(loaded_4) == 4
    counts_4 = Counter([p.category for p in loaded_4])
    assert counts_4["code"] == 2
    assert counts_4["logic"] == 1
    assert counts_4["math"] == 1


def test_load_suite_uneven_categories(tmp_path: Path) -> None:
    """Test distribution when some categories are exhausted."""
    prompt_file = tmp_path / "test_prompts_uneven.jsonl"

    # 5 math, 1 code
    prompts = []
    for i in range(5):
        prompts.append({"id": f"math-{i}", "text": f"math {i}", "category": "math"})
    prompts.append({"id": "code-0", "text": "code 0", "category": "code"})

    prompt_file.write_text("\n".join(json.dumps(p) for p in prompts))

    # Request 4 prompts.
    # Sorted categories: code, math
    # Round 1: code-0, math-0
    # Round 2: (code exhausted), math-1
    # Round 3: math-2
    loaded = load_suite(prompt_file, num_prompts=4)

    assert len(loaded) == 4
    counts = {p.category: 0 for p in loaded}
    for p in loaded:
        counts[p.category] += 1

    assert counts["code"] == 1
    assert counts["math"] == 3


def test_load_suite_no_limit(tmp_path: Path) -> None:
    """Test that load_suite returns all prompts if no limit is provided."""
    prompt_file = tmp_path / "test_prompts_all.jsonl"
    prompts = [{"id": "1", "text": "t1", "category": "a"}, {"id": "2", "text": "t2", "category": "b"}]
    prompt_file.write_text("\n".join(json.dumps(p) for p in prompts))

    loaded = load_suite(prompt_file)
    assert len(loaded) == 2
