"""
run_tests.py
------------
Universal test runner that executes unit tests for dataset integrity, scoring,
and the Streamlit backend using Python's built-in assertions.
"""

import os
import sys
import tempfile
import csv
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
TESTS_DIR = os.path.join(BASE_DIR, "tests")

for p in [SCRIPTS_DIR, TESTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from validate_dataset import (
    validate_prompts_file,
    validate_responses_file,
    REQUIRED_PROMPT_COLUMNS,
    REQUIRED_RESPONSE_COLUMNS
)
from score_responses import (
    process_scoring,
    parse_score
)
from streamlit_backend import (
    compute_pair_evaluation,
    recalculate_scored_and_summary,
    load_responses,
    get_default_paths,
    create_in_memory_audit_zip
)


def run_all_tests():
    print("=" * 70)
    print("UNIT TEST SUITE — CAPSTONE 16 RESPONSIBLE AI AUDIT & STREAMLIT BACKEND")
    print("=" * 70)

    test_results = []

    def record_test(name, passed, detail=""):
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name} {detail}")
        test_results.append((name, passed, detail))

    # Test 1: Valid Baseline Prompts
    try:
        prompts_path = os.path.join(BASE_DIR, "data", "test_prompts.csv")
        is_valid, issues, stats = validate_prompts_file(prompts_path)
        record_test("test_prompts_file_valid_baseline", is_valid and stats["total_rows"] == 60 and stats["unique_pairs"] == 30)
    except Exception as e:
        record_test("test_prompts_file_valid_baseline", False, str(e))

    # Test 2: Missing Required Columns
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tf:
            writer = csv.writer(tf)
            writer.writerow(["prompt_id", "prompt_text"])
            writer.writerow(["P01", "Text"])
            t_path = tf.name
        is_valid, issues, stats = validate_prompts_file(t_path)
        os.remove(t_path)
        record_test("test_missing_required_columns_rejected", (not is_valid) and any("Missing required columns" in iss for iss in issues))
    except Exception as e:
        record_test("test_missing_required_columns_rejected", False, str(e))

    # Test 3: Duplicate Prompt IDs
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tf:
            writer = csv.writer(tf)
            writer.writerow(REQUIRED_PROMPT_COLUMNS)
            writer.writerow(["P01", "PAIR_01", "baseline", "A", "Names", "Obj", "Prompt 1", "Attr", "Val1", "Val2", "Exp", "Crit", "Low", "Active", ""])
            writer.writerow(["P01", "PAIR_01", "comparison", "A", "Names", "Obj", "Prompt 2", "Attr", "Val2", "Val1", "Exp", "Crit", "Low", "Active", ""])
            t_path = tf.name
        is_valid, issues, stats = validate_prompts_file(t_path)
        os.remove(t_path)
        record_test("test_duplicate_prompt_ids_rejected", (not is_valid) and any("Duplicate prompt_id" in iss for iss in issues))
    except Exception as e:
        record_test("test_duplicate_prompt_ids_rejected", False, str(e))

    # Test 4: Incorrect Prompt Count per Pair
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tf:
            writer = csv.writer(tf)
            writer.writerow(REQUIRED_PROMPT_COLUMNS)
            writer.writerow(["P01_A", "PAIR_01", "baseline", "A", "Names", "Obj", "Prompt 1", "Attr", "Val1", "Val2", "Exp", "Crit", "Low", "Active", ""])
            t_path = tf.name
        is_valid, issues, stats = validate_prompts_file(t_path)
        os.remove(t_path)
        record_test("test_incorrect_prompt_count_per_pair", (not is_valid) and any("has 1 prompts" in iss for iss in issues))
    except Exception as e:
        record_test("test_incorrect_prompt_count_per_pair", False, str(e))

    # Test 5: Empty Input Files
    try:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as tf:
            t_path = tf.name
        is_valid, issues, stats = validate_prompts_file(t_path)
        os.remove(t_path)
        record_test("test_empty_input_file_handled", (not is_valid) and stats.get("total_rows", 0) == 0)
    except Exception as e:
        record_test("test_empty_input_file_handled", False, str(e))

    # Test 6: Rubric Score Parsing & Bounds [0, 4]
    try:
        b1 = (parse_score("0") == 0.0 and parse_score("4") == 4.0 and parse_score("2.5") == 2.5)
        b2 = (parse_score("") is None and parse_score(None) is None)
        out_of_bounds = False
        try:
            parse_score("5.0")
        except ValueError:
            out_of_bounds = True
        record_test("test_rubric_bounds_and_parsing", b1 and b2 and out_of_bounds)
    except Exception as e:
        record_test("test_rubric_bounds_and_parsing", False, str(e))

    # Test 7: Paired Difference Calculations
    try:
        config = {"thresholds": {"flag_disparity_delta": 1.0, "safety_critical_threshold": 2.0, "factual_critical_threshold": 2.0}}
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as in_f, \
             tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as out_f:
            writer = csv.writer(in_f)
            writer.writerow(REQUIRED_RESPONSE_COLUMNS)
            writer.writerow(["PAIR_TEST", "PTEST_BASE", "baseline", "A", "Attr", "ValA", "Bot", "v1", "2026-10-29", "Base Prompt", "Resp A", "SYNTHETIC_NOT_OBSERVED", "4", "4", "4", "4", "4", "Note", "Safe", "Review"])
            writer.writerow(["PAIR_TEST", "PTEST_COMP", "comparison", "A", "Attr", "ValB", "Bot", "v1", "2026-10-29", "Comp Prompt", "Resp B", "SYNTHETIC_NOT_OBSERVED", "2", "2", "2", "2", "2", "Note", "Safe", "Review"])
            in_p = in_f.name
            out_p = out_f.name

        scored = process_scoring(in_p, out_p, config)
        os.remove(in_p)
        os.remove(out_p)

        rec = scored[0]
        math_correct = (rec["base_overall_mean"] == 4.0 and rec["comp_overall_mean"] == 2.0 and
                        rec["delta_overall_mean"] == 2.0 and rec["abs_overall_delta"] == 2.0 and
                        rec["manual_review_required"] == "YES")
        record_test("test_paired_difference_calculation", math_correct)
    except Exception as e:
        record_test("test_paired_difference_calculation", False, str(e))

    # Test 8: Missing Score Handling
    try:
        config = {"thresholds": {"flag_disparity_delta": 1.0, "safety_critical_threshold": 2.0, "factual_critical_threshold": 2.0}}
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as in_f, \
             tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as out_f:
            writer = csv.writer(in_f)
            writer.writerow(REQUIRED_RESPONSE_COLUMNS)
            writer.writerow(["PAIR_MISS", "PMISS_BASE", "baseline", "A", "Attr", "ValA", "Bot", "v1", "2026-10-29", "Prompt", "Resp", "SYNTHETIC_NOT_OBSERVED", "4", "", "4", "", "", "Note", "Safe", "Review"])
            writer.writerow(["PAIR_MISS", "PMISS_COMP", "comparison", "A", "Attr", "ValB", "Bot", "v1", "2026-10-29", "Prompt", "Resp", "SYNTHETIC_NOT_OBSERVED", "", "", "", "", "", "Note", "Safe", "Review"])
            in_p = in_f.name
            out_p = out_f.name

        scored = process_scoring(in_p, out_p, config)
        os.remove(in_p)
        os.remove(out_p)

        rec = scored[0]
        missing_handled = (rec["base_overall_mean"] == 4.0 and rec["comp_overall_mean"] is None and rec["delta_overall_mean"] is None)
        record_test("test_missing_scores_handled_safely", missing_handled)
    except Exception as e:
        record_test("test_missing_scores_handled_safely", False, str(e))

    # Test 9: Official Sample Responses Validation
    try:
        sample_path = os.path.join(BASE_DIR, "data", "chatbot_responses_sample.csv")
        is_valid, issues, stats = validate_responses_file(sample_path)
        record_test("test_sample_responses_dataset_valid", is_valid and stats["synthetic_rows"] == 60)
    except Exception as e:
        record_test("test_sample_responses_dataset_valid", False, str(e))

    # Test 10: Streamlit Dynamic Recalculation
    try:
        paths = get_default_paths(BASE_DIR)
        resp_df = load_responses(paths["sample_responses_csv"])
        s_df, g_df = recalculate_scored_and_summary(resp_df, {"thresholds": {"flag_disparity_delta": 1.0}})
        record_test("test_streamlit_dynamic_recalculation", len(s_df) == 30 and len(g_df) == 6)
    except Exception as e:
        record_test("test_streamlit_dynamic_recalculation", False, str(e))

    # Test 11: Streamlit Disparity Flagging Logic
    try:
        base_r = {"pair_id": "P1", "category": "A", "score_fairness_consistency": 4.0, "score_factual_reliability": 4.0, "score_safety": 4.0, "score_respect_inclusiveness": 4.0, "score_transparency_uncertainty": 4.0}
        comp_r = {"pair_id": "P1", "category": "A", "score_fairness_consistency": 2.0, "score_factual_reliability": 4.0, "score_safety": 4.0, "score_respect_inclusiveness": 3.0, "score_transparency_uncertainty": 4.0}
        res = compute_pair_evaluation(base_r, comp_r, flag_delta_thresh=1.0)
        record_test("test_streamlit_flagging_logic", res["manual_review_required"] == "YES" and res["delta_fairness_consistency"] == 2.0)
    except Exception as e:
        record_test("test_streamlit_flagging_logic", False, str(e))

    # Test 12: Streamlit In-Memory Zip Packaging
    try:
        paths = get_default_paths(BASE_DIR)
        resp_df = load_responses(paths["sample_responses_csv"])
        s_df, g_df = recalculate_scored_and_summary(resp_df, {"thresholds": {"flag_disparity_delta": 1.0}})
        zip_bytes = create_in_memory_audit_zip(BASE_DIR, resp_df, s_df, g_df)
        record_test("test_streamlit_in_memory_zip_packaging", len(zip_bytes) > 2000)
    except Exception as e:
        record_test("test_streamlit_in_memory_zip_packaging", False, str(e))

    print("=" * 70)
    all_passed = all(r[1] for r in test_results)
    passed_count = sum(1 for r in test_results if r[1])
    total_count = len(test_results)
    print(f"TEST EXECUTION SUMMARY: {passed_count}/{total_count} PASSED ({'ALL PASS' if all_passed else 'FAILURES DETECTED'})")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
