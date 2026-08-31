"""
Covers v22b.4 (prompts.py section 7): a science-claim/source-type
strictness hotfix on top of v22b3. Motivated by a real, trace-verified gap:
v22b3's Italian rerun fixed the IT125 explanation leak (verified working),
but IT125's counter_narrative said "La scienza conferma che l'orientamento
sessuale non e una scelta..." ("Science confirms sexual orientation is not
a choice"), licensed only by a source_type="cultural" passage whose own
text happened to say "...supported by science and psychology." v22b2/
v22b3's Rule 5 had an exception for exactly this ("unless the passage's own
text contains the claim"), which was too permissive - it let a cultural
passage's incidental mention of "science" license a citation-style claim,
and let a general claim ("natural expression of human diversity") get
rewritten as a stronger, more specific one ("not a choice") the evidence
never said. v22b4 removes that exception for citation framing and adds an
explicit no-strengthening rule.

v22b4 does NOT replace v20/v22b/v22b1/v22b2/v22b3 - all six remain
independently selectable. v22b3's EXPLANATION-FIELD RULES are reused
UNCHANGED (they were trace-verified to work - see docs/07_EVALUATION_STRATEGY.md's
"v22b.4" section). These tests check the prompt TEXT asks for the right
things; live model verification needs a real model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prompts
from main import _tag_filename_for_final_cn_style

_SAMPLE_JUDGE_PLAN = {
    "selected_language": "it", "core_claim_to_counter": "gay people are sick",
    "recommended_strategy": ["factual_correction"],
    "approved_evidence": [{"source_id": "s1", "title": "t",
                            "passage": "homosexuality is natural and is supported by science and psychology.",
                            "retrieval_score": 0.9, "source_type": "cultural", "language": "it"}],
    "rejected_content": [], "cultural_guidance": ["Italian culture values family and dignity."],
    "safety_guidance": [], "final_response_plan": "Rebut using the approved evidence.",
}


def _v22b4(language="it", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b4")


def _v22b3(language="it", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b3")


def _v22b2(language="it", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v22b2")


def _v20(language="it", region_context="unknown"):
    return prompts.build_final_cn_prompt("test comment", language, _SAMPLE_JUDGE_PLAN,
                                          region_context, style="v20")


def _normalized(text: str) -> str:
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# v22b4 exists and is selectable / v22b3 remains unchanged / filename suffix
# ---------------------------------------------------------------------------
def test_v22b4_is_selectable():
    text = _v22b4()
    assert "NO CULTURAL-PASSAGE EXCEPTION" in text


def test_v22b3_remains_unchanged_and_v20_still_default():
    v22b3_before = _v22b3()  # captured fresh each call from the unchanged v22b3 builder
    assert "NO CULTURAL-PASSAGE EXCEPTION" not in v22b3_before
    assert "EXPLANATION-FIELD RULES" in v22b3_before
    v20, v22b2, v22b3, v22b4 = _v20(), _v22b2(), _v22b3(), _v22b4()
    assert len({v20, v22b2, v22b3, v22b4}) == 4
    assert prompts.build_final_cn_prompt("test comment", "it", _SAMPLE_JUDGE_PLAN, "unknown") == v20


def test_filename_suffix_for_v22b4_is_distinct():
    names = {s: _tag_filename_for_final_cn_style("it", s)
             for s in ("v20", "v22b", "v22b1", "v22b2", "v22b3", "v22b4")}
    assert names["v20"] == "it"
    assert names["v22b4"] == "it-finalcnv22b4"
    assert len(set(names.values())) == 6


# ---------------------------------------------------------------------------
# cultural/culturale source_type cannot support "la scienza conferma"
# ---------------------------------------------------------------------------
def test_cultural_source_type_cannot_license_science_confirms():
    text = _normalized(_v22b4())
    assert "la scienza conferma" in text
    assert ('EVEN IF that passage\'s own wording happens to mention "science"' in text
            or "even if that passage's own wording happens to mention" in text.lower())


def test_culturale_explicitly_listed_as_non_matching_source_type():
    text = _normalized(_v22b4())
    assert '"culturale"' in text
    assert "that wording inside a cultural passage does not upgrade it to" in text.lower() \
        or "does not upgrade it to" in text.lower()


def test_science_claim_requires_exact_fact_or_web_source_type():
    text = _normalized(_v22b4())
    assert 'source_type is EXACTLY the string "fact" or EXACTLY the string "web"' in text


# ---------------------------------------------------------------------------
# "natural expression of human diversity" cannot become "not a choice"
# ---------------------------------------------------------------------------
def test_no_strengthening_rule_has_the_exact_example():
    text = _normalized(_v22b4())
    assert "natural expression of human diversity" in text
    assert "is not a choice" in text
    assert "DIFFERENT claims" in text


# ---------------------------------------------------------------------------
# explanation rules from v22b3 remain present (unchanged, not weakened)
# ---------------------------------------------------------------------------
def test_v22b3_explanation_rules_carried_over_verbatim():
    text = _normalized(_v22b4())
    assert "EXPLANATION-FIELD RULES" in text
    assert "method-level summary only, not a second" in text.lower()
    assert "approccio culturale italiano" in text
    assert "cultura italiana" in text
    assert "scienza e storia" in text


# ---------------------------------------------------------------------------
# Italian naturalness rule includes the grammar fix
# ---------------------------------------------------------------------------
def test_italian_grammar_fix_present():
    text = _v22b4(language="it")
    assert "dal proprio orientamento sessuale" in text
    assert "dalla propria orientamento sessuale" in text  # cited as the WRONG example to avoid


# ---------------------------------------------------------------------------
# weak-evidence fallback uses the exact IT125-style example given
# ---------------------------------------------------------------------------
def test_it125_style_safe_fallback_example_present():
    text = _v22b4(language="it")
    assert "Ogni bambino merita amore e accettazione" in text
    assert "non diminuisce il valore di una persona" in text


if __name__ == "__main__":
    test_v22b4_is_selectable()
    test_v22b3_remains_unchanged_and_v20_still_default()
    test_filename_suffix_for_v22b4_is_distinct()
    test_cultural_source_type_cannot_license_science_confirms()
    test_culturale_explicitly_listed_as_non_matching_source_type()
    test_science_claim_requires_exact_fact_or_web_source_type()
    test_no_strengthening_rule_has_the_exact_example()
    test_v22b3_explanation_rules_carried_over_verbatim()
    test_italian_grammar_fix_present()
    test_it125_style_safe_fallback_example_present()
    print("test_final_cn_prompt_v22b4.py: ALL PASSED")
