from infer_check.analysis.clustering import cluster_failures, summarize_clusters
from infer_check.types import ComparisonResult, InferenceResult, Prompt


def test_cluster_failures_by_category() -> None:
    prompts = {
        "p1": Prompt(id="p1", text="A", category="coding", max_tokens=10),
        "p2": Prompt(id="p2", text="B", category="math", max_tokens=10),
    }

    comp1 = ComparisonResult(
        baseline=InferenceResult(
            prompt_id="p1", backend_name="base", model_id="m", tokens=[], text="", latency_ms=1.0
        ),
        test=InferenceResult(
            prompt_id="p1",
            backend_name="test",
            model_id="m",
            tokens=[],
            text="diff",
            latency_ms=1.0,
        ),
        text_similarity=0.1,
        is_failure=True,
        failure_reason="fail",
        token_divergence_index=0,
        kl_divergence=None,
        metadata={},
    )

    # Simulate a successful prompt to ensure it is ignored
    comp2_success = ComparisonResult(
        baseline=InferenceResult(
            prompt_id="p2", backend_name="base", model_id="m", tokens=[], text="", latency_ms=1.0
        ),
        test=InferenceResult(
            prompt_id="p2", backend_name="test", model_id="m", tokens=[], text="", latency_ms=1.0
        ),
        text_similarity=1.0,
        is_failure=False,
        failure_reason=None,
        token_divergence_index=None,
        kl_divergence=None,
        metadata={},
    )

    clusters = cluster_failures([comp1, comp2_success], prompts)
    assert "category:coding" in clusters
    assert len(clusters["category:coding"]) == 1
    assert "category:math" not in clusters  # success ignored


def test_cluster_failures_by_length() -> None:
    short_text = "short text"
    long_text = " ".join(["word"] * 600)

    prompts = {
        "short_p": Prompt(id="short_p", text=short_text, category="c", max_tokens=10),
        "long_p": Prompt(id="long_p", text=long_text, category="c", max_tokens=10),
    }

    base_res = InferenceResult(
        prompt_id="short_p", backend_name="b", model_id="m", text="", latency_ms=1, tokens=[]
    )
    test_res = InferenceResult(
        prompt_id="short_p", backend_name="t", model_id="m", text="diff", latency_ms=1, tokens=[]
    )
    comp1 = ComparisonResult(
        baseline=base_res,
        test=test_res,
        text_similarity=0.1,
        is_failure=True,
        metadata={},
        token_divergence_index=0,
    )

    base_res_long = InferenceResult(
        prompt_id="long_p", backend_name="b", model_id="m", text="", latency_ms=1, tokens=[]
    )
    test_res_long = InferenceResult(
        prompt_id="long_p", backend_name="t", model_id="m", text="diff", latency_ms=1, tokens=[]
    )
    comp2 = ComparisonResult(
        baseline=base_res_long,
        test=test_res_long,
        text_similarity=0.1,
        is_failure=True,
        metadata={},
        token_divergence_index=0,
    )

    clusters = cluster_failures([comp1, comp2], prompts)
    assert "length:short" in clusters
    assert "length:long" in clusters


def test_cluster_failures_by_onset() -> None:
    prompts = {"p1": Prompt(id="p1", text="x", category="c", max_tokens=10)}
    base_res = InferenceResult(
        prompt_id="p1", backend_name="b", model_id="m", text="", latency_ms=1, tokens=[]
    )
    test_res = InferenceResult(
        prompt_id="p1", backend_name="t", model_id="m", text="diff", latency_ms=1, tokens=[]
    )

    # Early onset (diverged at index 5)
    comp1 = ComparisonResult(
        baseline=base_res,
        test=test_res,
        text_similarity=0.1,
        is_failure=True,
        token_divergence_index=5,
        metadata={},
    )
    # Late onset (diverged at index 60)
    comp2 = ComparisonResult(
        baseline=base_res,
        test=test_res,
        text_similarity=0.1,
        is_failure=True,
        token_divergence_index=60,
        metadata={},
    )

    clusters = cluster_failures([comp1, comp2], prompts)
    assert "onset:early" in clusters
    assert "onset:late" in clusters


def test_summarize_clusters() -> None:
    prompts = {
        "p1": Prompt(id="p1", text="c", category="coding", max_tokens=10),
        "p2": Prompt(id="p2", text="c", category="coding", max_tokens=10),
        "p3": Prompt(id="p3", text="m", category="math", max_tokens=10),
    }

    base_res = InferenceResult(
        prompt_id="p1", backend_name="b", model_id="m", text="", latency_ms=1, tokens=[]
    )
    test_res = InferenceResult(
        prompt_id="p1", backend_name="t", model_id="m", text="diff", latency_ms=1, tokens=[]
    )
    c1 = ComparisonResult(
        baseline=base_res,
        test=test_res,
        text_similarity=0.1,
        is_failure=True,
        token_divergence_index=0,
        metadata={},
    )

    base_res2 = InferenceResult(
        prompt_id="p2", backend_name="b", model_id="m", text="", latency_ms=1, tokens=[]
    )
    test_res2 = InferenceResult(
        prompt_id="p2", backend_name="t", model_id="m", text="diff", latency_ms=1, tokens=[]
    )
    c2 = ComparisonResult(
        baseline=base_res2,
        test=test_res2,
        text_similarity=0.1,
        is_failure=True,
        token_divergence_index=0,
        metadata={},
    )

    # Just setting distinct IDs to trick 'id()' call in summarize if needed,
    # but the instances are already unique.

    clusters = cluster_failures([c1, c2], prompts)
    summary = summarize_clusters(clusters)

    assert "Failures concentrated in:" in summary
    assert "coding prompts" in summary
    assert "100%" in summary  # both failed coding prompts out of 2 total failures
