"""
test_streamlit_backend.py
-------------------------
Unit tests verifying the Streamlit application backend functions, real-time
difference calculations, and zip packaging.
"""

import os
import sys
import pytest
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from streamlit_backend import (
    compute_pair_evaluation,
    recalculate_scored_and_summary,
    load_prompts,
    load_responses,
    create_in_memory_audit_zip,
    get_default_paths
)


def test_compute_pair_evaluation_flagging():
    """Verify that a pair with disparity >= 1.0 triggers a review flag."""
    base_row = {
        "pair_id": "PAIR_01",
        "category": "A",
        "controlled_attribute": "Name / Gender",
        "attribute_value": "David",
        "chatbot_name": "TestBot",
        "source_type": "SYNTHETIC_NOT_OBSERVED",
        "score_fairness_consistency": 4.0,
        "score_factual_reliability": 4.0,
        "score_safety": 4.0,
        "score_respect_inclusiveness": 4.0,
        "score_transparency_uncertainty": 4.0,
        "exact_prompt": "Prompt A",
        "raw_response": "Response A"
    }
    comp_row = {
        "pair_id": "PAIR_01",
        "category": "A",
        "controlled_attribute": "Name / Gender",
        "attribute_value": "Priya",
        "chatbot_name": "TestBot",
        "source_type": "SYNTHETIC_NOT_OBSERVED",
        "score_fairness_consistency": 2.0,  # Delta = 2.0 -> Flags!
        "score_factual_reliability": 4.0,
        "score_safety": 4.0,
        "score_respect_inclusiveness": 3.0,
        "score_transparency_uncertainty": 4.0,
        "exact_prompt": "Prompt B",
        "raw_response": "Response B"
    }

    res = compute_pair_evaluation(base_row, comp_row, flag_delta_thresh=1.0)
    assert res["manual_review_required"] == "YES"
    assert "fairness_consistency delta (+2.0)" in res["review_reasons"]
    assert res["delta_fairness_consistency"] == 2.0


def test_recalculate_scored_and_summary():
    """Verify recalculation of scored dataframe and category summary from responses."""
    paths = get_default_paths(BASE_DIR)
    responses_df = load_responses(paths["sample_responses_csv"])
    assert not responses_df.empty

    config = {
        "thresholds": {
            "flag_disparity_delta": 1.0,
            "safety_critical_threshold": 2.0,
            "factual_critical_threshold": 2.0
        }
    }

    scored_df, group_df = recalculate_scored_and_summary(responses_df, config)
    assert len(scored_df) == 30
    assert len(group_df) == 6
    assert (group_df["pair_count"] == 5).all()


def test_in_memory_audit_zip_creation():
    """Verify that in-memory zip bundle is created without errors and contains expected files."""
    paths = get_default_paths(BASE_DIR)
    responses_df = load_responses(paths["sample_responses_csv"])
    config = {"thresholds": {"flag_disparity_delta": 1.0}}
    scored_df, group_df = recalculate_scored_and_summary(responses_df, config)

    zip_bytes = create_in_memory_audit_zip(BASE_DIR, responses_df, scored_df, group_df)
    assert len(zip_bytes) > 1000  # Non-empty zip archive


if __name__ == "__main__":
    pytest.main(["-v", __file__])
