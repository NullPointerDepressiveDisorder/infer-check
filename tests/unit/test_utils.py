from infer_check.utils import sanitize_filename


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
