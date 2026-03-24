"""Tests for infer_check.analysis.answer_extract."""

from __future__ import annotations

import pytest

from infer_check.analysis.answer_extract import (
    answers_match,
    compute_flip_rate,
    extract_answer,
)

# ═════════════════════════════════════════════════════════════════════
# Numeric extraction
# ═════════════════════════════════════════════════════════════════════


class TestNumericExtraction:
    def test_simple_integer(self) -> None:
        ans = extract_answer("The answer is 42.", "arithmetic")
        assert ans.strategy == "numeric"
        assert ans.value == "42"
        assert ans.confidence > 0.5

    def test_decimal(self) -> None:
        ans = extract_answer("0.1 + 0.2 = 0.30000000000000004", "floating_point")
        assert ans.value == "0.30000000000000004"

    def test_scientific_notation(self) -> None:
        ans = extract_answer("The result is approximately 1.7e308", "large_numbers")
        assert ans.value == "1.7e308"

    def test_comma_separated_number(self) -> None:
        ans = extract_answer("The conversion gives 1,234,567.89 yen.", "precision")
        assert ans.value == "1234567.89"

    def test_negative_number(self) -> None:
        ans = extract_answer("The answer is -273.15", "precision")
        assert ans.value == "-273.15"

    def test_last_number_wins(self) -> None:
        ans = extract_answer(
            "First I calculate 100, then 200, final answer is 300.",
            "arithmetic",
        )
        assert ans.value == "300"

    def test_no_number_found(self) -> None:
        ans = extract_answer("I cannot compute this.", "arithmetic")
        assert ans.value == ""
        assert ans.confidence == 0.0

    def test_word_problem_category(self) -> None:
        ans = extract_answer("They meet at 4:30 PM, after 135 minutes.", "word_problem")
        assert ans.strategy == "numeric"
        assert ans.value == "135"


# ═════════════════════════════════════════════════════════════════════
# Boolean extraction
# ═════════════════════════════════════════════════════════════════════


class TestBooleanExtraction:
    def test_yes_answer(self) -> None:
        ans = extract_answer("After checking, yes, all bloops are lazzles.", "logic")
        assert ans.strategy == "boolean"
        assert ans.value == "yes"

    def test_no_answer(self) -> None:
        ans = extract_answer("No, that is not correct.", "logic")
        assert ans.value == "no"

    def test_true_normalised_to_yes(self) -> None:
        ans = extract_answer("This statement is True.", "edge_case")
        assert ans.value == "yes"

    def test_definitely_normalised(self) -> None:
        ans = extract_answer("Yes, definitely, all bloops are lazzles.", "logic")
        assert ans.value == "yes"

    def test_last_keyword_wins(self) -> None:
        ans = extract_answer("It might seem like yes, but actually no.", "logic")
        assert ans.value == "no"

    def test_no_boolean_found(self) -> None:
        ans = extract_answer("The result is indeterminate.", "logic")
        assert ans.value == ""
        assert ans.confidence == 0.0


# ═════════════════════════════════════════════════════════════════════
# Code extraction
# ═════════════════════════════════════════════════════════════════════


class TestCodeExtraction:
    def test_fenced_block(self) -> None:
        text = "Here's the solution:\n```python\ndef add(a, b):\n    return a + b\n```"
        ans = extract_answer(text, "python")
        assert ans.strategy == "code"
        assert "def add(a, b):" in ans.value
        assert ans.confidence == 0.9

    def test_unfenced_code(self) -> None:
        text = "def fib(n):\n    if n <= 1:\n        return n\n    return fib(n-1) + fib(n-2)"
        ans = extract_answer(text, "python")
        assert "def fib" in ans.value

    def test_no_code_found(self) -> None:
        ans = extract_answer("I'm not sure how to solve this.", "python")
        assert ans.confidence <= 0.5


# ═════════════════════════════════════════════════════════════════════
# JSON extraction
# ═════════════════════════════════════════════════════════════════════


class TestJsonExtraction:
    def test_fenced_json(self) -> None:
        text = '```json\n{"name": "Alice", "age": 30}\n```'
        ans = extract_answer(text, "json")
        assert ans.strategy == "json"
        assert '"age":30' in ans.value
        assert '"name":"Alice"' in ans.value
        assert ans.confidence == 0.95

    def test_inline_json(self) -> None:
        text = 'The user profile is: {"name": "Bob", "email": "b@x.com"}'
        ans = extract_answer(text, "json")
        assert ans.strategy == "json"
        assert ans.confidence == 0.95

    def test_json_array(self) -> None:
        text = 'Here: [{"id": 1}, {"id": 2}]'
        ans = extract_answer(text, "json")
        assert "[" in ans.value
        assert ans.confidence == 0.95

    def test_invalid_json(self) -> None:
        text = "This is not json {broken"
        ans = extract_answer(text, "json")
        assert ans.value == ""
        assert ans.confidence == 0.0

    def test_json_key_order_normalised(self) -> None:
        """Different key order → same canonical output."""
        a = extract_answer('{"b": 2, "a": 1}', "json")
        b = extract_answer('{"a": 1, "b": 2}', "json")
        assert a.value == b.value


# ═════════════════════════════════════════════════════════════════════
# answers_match
# ═════════════════════════════════════════════════════════════════════


class TestAnswersMatch:
    def test_identical_numeric(self) -> None:
        a = extract_answer("42", "arithmetic")
        b = extract_answer("The answer is 42.", "arithmetic")
        assert answers_match(a, b)

    def test_different_numeric(self) -> None:
        a = extract_answer("42", "arithmetic")
        b = extract_answer("43", "arithmetic")
        assert not answers_match(a, b)

    def test_identical_boolean(self) -> None:
        a = extract_answer("Yes, definitely.", "logic")
        b = extract_answer("That is correct.", "logic")
        assert answers_match(a, b)

    def test_different_boolean(self) -> None:
        a = extract_answer("Yes", "logic")
        b = extract_answer("No", "logic")
        assert not answers_match(a, b)

    def test_identical_json(self) -> None:
        a = extract_answer('{"a":1,"b":2}', "json")
        b = extract_answer('{"b":2,"a":1}', "json")
        assert answers_match(a, b)

    def test_code_similarity_match(self) -> None:
        a = extract_answer("```python\ndef f(x):\n    return x+1\n```", "python")
        b = extract_answer("```python\ndef f(x):\n    return x + 1\n```", "python")
        assert answers_match(a, b)

    def test_one_empty_extraction_is_flip(self) -> None:
        a = extract_answer("42", "arithmetic")
        b = extract_answer("I cannot compute this.", "arithmetic")
        assert not answers_match(a, b)

    def test_both_empty_extraction_not_flip(self) -> None:
        a = extract_answer("dunno", "arithmetic")
        b = extract_answer("no idea", "arithmetic")
        assert answers_match(a, b)


# ═════════════════════════════════════════════════════════════════════
# compute_flip_rate
# ═════════════════════════════════════════════════════════════════════


class TestComputeFlipRate:
    def test_no_pairs(self) -> None:
        rate, details = compute_flip_rate([])
        assert rate == 0.0
        assert details == []

    def test_all_match(self) -> None:
        pairs = [
            ("p1", "arithmetic", "42", "The answer is 42"),
            ("p2", "logic", "Yes, correct.", "Yes"),
        ]
        rate, details = compute_flip_rate(pairs)
        assert rate == 0.0
        assert len(details) == 2
        assert all(not d.flipped for d in details)

    def test_all_flipped(self) -> None:
        pairs = [
            ("p1", "arithmetic", "42", "99"),
            ("p2", "logic", "Yes", "No"),
        ]
        rate, details = compute_flip_rate(pairs)
        assert rate == 1.0
        assert all(d.flipped for d in details)

    def test_partial_flip(self) -> None:
        pairs = [
            ("p1", "arithmetic", "42", "42"),
            ("p2", "arithmetic", "42", "99"),
            ("p3", "logic", "Yes", "Yes"),
            ("p4", "logic", "Yes", "No"),
        ]
        rate, details = compute_flip_rate(pairs)
        assert rate == pytest.approx(0.5)
        flipped_ids = {d.prompt_id for d in details if d.flipped}
        assert flipped_ids == {"p2", "p4"}

    def test_details_contain_extracted_answers(self) -> None:
        pairs = [("p1", "arithmetic", "100", "200")]
        _, details = compute_flip_rate(pairs)
        assert len(details) == 1
        d = details[0]
        assert d.answer_a.value == "100"
        assert d.answer_b.value == "200"
        assert d.answer_a.strategy == "numeric"


# ═════════════════════════════════════════════════════════════════════
# Fallback / raw strategy
# ═════════════════════════════════════════════════════════════════════


class TestRawFallback:
    def test_unknown_category_uses_raw(self) -> None:
        ans = extract_answer("Some generic response", "unknown_cat")
        assert ans.strategy == "raw"
        assert ans.confidence == 0.3

    def test_general_category_uses_raw(self) -> None:
        ans = extract_answer("Hello world", "general")
        assert ans.strategy == "raw"

    def test_new_categories_mapped_correctly(self) -> None:
        # New categories from quant-sensitive.jsonl and their expected strategies
        category_mapping = {
            "multi_digit_arithmetic": "numeric",
            "precision_numerics": "numeric",
            "large_number_reasoning": "numeric",
            "logical_puzzle": "numeric",
            "algebraic_reasoning": "numeric",
            "precise_syntax": "code",
            "code_translation": "code",
            "long_chain_of_thought": "raw",
        }

        for cat, expected_strategy in category_mapping.items():
            ans = extract_answer("The answer is 42.", category=cat)
            assert ans.strategy == expected_strategy, f"Category {cat} should use {expected_strategy} strategy"
