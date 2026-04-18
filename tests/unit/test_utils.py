from unittest.mock import MagicMock

from infer_check.utils import format_prompt, sanitize_filename, strip_thinking_tokens


def test_sanitize_filename_basic() -> None:
    assert sanitize_filename("basic") == "basic"
    assert sanitize_filename("label with spaces") == "label with spaces"
    assert sanitize_filename("path/separator") == "path_separator"
    assert sanitize_filename("back\\slash") == "back_slash"


def test_sanitize_filename_unsafe_chars() -> None:
    assert sanitize_filename("file:name*with?unsafe<chars>") == "file_name_with_unsafe_chars"


def test_sanitize_filename_underscores() -> None:
    assert sanitize_filename("many___underscores") == "many_underscores"
    assert sanitize_filename("__leading_and_trailing__") == "leading_and_trailing"


def test_sanitize_filename_empty_or_unsafe_only() -> None:
    assert sanitize_filename("") == "model"
    assert sanitize_filename("/") == "model"
    assert sanitize_filename("___") == "model"


def test_sanitize_filename_windows_trailing_dot() -> None:
    # Windows doesn't allow trailing dots or spaces
    assert sanitize_filename("model.") == "model"
    assert sanitize_filename("model ") == "model"
    assert sanitize_filename("model. ") == "model"
    assert sanitize_filename("model .") == "model"


def test_sanitize_filename_windows_reserved_names() -> None:
    # Windows reserved names: CON, PRN, AUX, NUL, COM1-9, LPT1-9
    # These should be sanitized to avoid issues on Windows
    reserved_names = ["CON", "PRN", "AUX", "NUL", "COM1", "COM9", "LPT1", "LPT9"]
    for name in reserved_names:
        # Case insensitive check
        assert sanitize_filename(name) == f"{name}_"
        assert sanitize_filename(name.lower()) == f"{name.lower()}_"


def test_sanitize_filename_combination() -> None:
    assert sanitize_filename("  __model.   ") == "model"
    assert sanitize_filename("my.model.name") == "my.model.name"  # dots in middle are fine


def _make_fake_tokenizer(accepted_kwargs: set[str] | None) -> MagicMock:
    """Build a tokenizer mock whose apply_chat_template accepts only selected kwargs.

    ``accepted_kwargs=None`` means the template rejects *every* extra kwarg
    (simulating a non-thinking model's template).
    """
    tok = MagicMock()
    tok.chat_template = "stub"

    def apply(messages, **kwargs):  # type: ignore[no-untyped-def]
        for key in ("enable_thinking", "thinking"):
            if key in kwargs:
                if accepted_kwargs is None or key not in accepted_kwargs:
                    raise TypeError(f"got unexpected kwarg {key!r}")
                return f"templated[{key}={kwargs[key]}]"
        return "templated"

    tok.apply_chat_template = MagicMock(side_effect=apply)
    return tok


def test_format_prompt_disable_thinking_qwen3_style() -> None:
    tok = _make_fake_tokenizer(accepted_kwargs={"enable_thinking"})
    out = format_prompt("hi", tokenizer=tok, disable_thinking=True)
    assert out == "templated[enable_thinking=False]"


def test_format_prompt_disable_thinking_deepseek_style() -> None:
    tok = _make_fake_tokenizer(accepted_kwargs={"thinking"})
    out = format_prompt("hi", tokenizer=tok, disable_thinking=True)
    assert out == "templated[thinking=False]"


def test_format_prompt_disable_thinking_non_thinking_model_falls_back() -> None:
    # Template rejects both flags — we still render a normal prompt.
    tok = _make_fake_tokenizer(accepted_kwargs=None)
    out = format_prompt("hi", tokenizer=tok, disable_thinking=True)
    assert out == "templated"


def test_format_prompt_default_does_not_pass_thinking_kwargs() -> None:
    tok = _make_fake_tokenizer(accepted_kwargs={"enable_thinking"})
    out = format_prompt("hi", tokenizer=tok)
    assert out == "templated"


def test_format_prompt_no_chat_template_returns_text() -> None:
    tok = MagicMock()
    tok.chat_template = None
    assert format_prompt("raw text", tokenizer=tok, disable_thinking=True) == "raw text"


def test_strip_thinking_tokens_ollama_trigger() -> None:
    # Ollama's gpt-oss uses <|think|> as a system-prompt trigger.
    assert strip_thinking_tokens("<|think|>solve x") == "solve x"
    assert strip_thinking_tokens("<|THINK|>hi") == "hi"


def test_strip_thinking_tokens_deepseek_wrapper() -> None:
    assert strip_thinking_tokens("<think>reasoning here</think>answer") == "answer"
    # Cross-line reasoning traces.
    assert strip_thinking_tokens("before<think>\nstep1\nstep2\n</think>after") == "beforeafter"
    # Stray unbalanced tags are also removed.
    assert strip_thinking_tokens("stray <think> tag") == "stray  tag"


def test_strip_thinking_tokens_leaves_normal_text_untouched() -> None:
    assert strip_thinking_tokens("Hello, world!") == "Hello, world!"


def test_format_prompt_strips_thinking_tokens_when_disabled() -> None:
    tok = _make_fake_tokenizer(accepted_kwargs={"enable_thinking"})
    format_prompt("<|think|>hi", tokenizer=tok, disable_thinking=True)
    # The message forwarded to the template must not carry the trigger token.
    args, _ = tok.apply_chat_template.call_args
    assert args[0] == [{"role": "user", "content": "hi"}]


def test_format_prompt_keeps_thinking_tokens_when_enabled() -> None:
    tok = _make_fake_tokenizer(accepted_kwargs={"enable_thinking"})
    format_prompt("<|think|>hi", tokenizer=tok, disable_thinking=False)
    args, _ = tok.apply_chat_template.call_args
    assert args[0] == [{"role": "user", "content": "<|think|>hi"}]
