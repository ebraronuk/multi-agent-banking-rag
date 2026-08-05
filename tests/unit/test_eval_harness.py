from __future__ import annotations

from evaluation.eval_harness import run_intent_eval


def test_rule_based_intent_classifier_clears_a_minimum_bar_on_the_eval_set() -> None:
    result = run_intent_eval()

    assert result.accuracy >= 0.85, f"misses: {result.misses}"
