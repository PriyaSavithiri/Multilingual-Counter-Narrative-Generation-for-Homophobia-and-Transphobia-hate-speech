"""
Durable, automated guard for the v22 unsafe-keyword redesign's core
constraint: this whole effort (Stages 1-10) is EVALUATION-ONLY and must
never change retrieval or generation behaviour (explicit user instruction).
Manual byte-diffs against the v20 zip were run after every stage during
development, but a manual diff doesn't survive into the next session or
guard against an accidental future edit - this pins a SHA-256 manifest of
every file on the actual generation path (agents, pipeline orchestration,
schemas, model calls, routing/safety) so a real change trips a hard test
failure instead of being silently missed.

If this test ever fails because one of these files was intentionally
changed (e.g. the upcoming Mistral final-CN prompt improvement work, which
WILL need to touch generation files), that's expected - update
_EXPECTED_HASHES to the new hash and note in the same commit/PR why the
generation path changed. A failure here must never be silenced without
that explicit acknowledgement, since the whole point is to make generation-
path drift visible rather than assumed away.

rag_pipeline.py is deliberately NOT in this manifest even though it's
retrieval-adjacent - unlike the files below, it DOES contain evaluation-
related code (_quality_filter's frozen-to-v20 unsafe-keyword check) mixed
in with retrieval code, so a whole-file hash would trip on legitimate
evaluation-only edits elsewhere in that file. Its generation-relevant
behavior (that _quality_filter reproduces v20's exact keep/drop decisions)
is instead pinned behaviorally in test_unsafe_keywords.py's
test_quality_filter_matches_v20_bare_substring_behavior_exactly and
test_quality_filter_still_drops_spanish_sin_by_design.

model_api.py's hash WAS deliberately updated (2026-08-11, twice) - the one
intentional exception exercised so far. Fix: Mistral-family tokenizers
(confirmed live: gghfez/Mistral-Small-3.2-24B-Instruct-hf) emit a
transformers warning about an incorrect split regex ("set
fix_mistral_regex=True to fix this"). The FIRST version of this fix passed
fix_mistral_regex=True into the single shared model_kwargs dict that
HuggingFacePipeline.from_model_id gives to BOTH AutoTokenizer.from_pretrained
AND AutoModelForCausalLM.from_pretrained - which crashed a real Colab run
with "MistralForCausalLM.__init__() got an unexpected keyword argument
'fix_mistral_regex'", since that's a tokenizer-only constructor argument.
The SECOND (current) version fixes this properly: HFTransformersClient._load()
now builds the tokenizer, model, and pipeline manually with separate kwargs
dicts (_hf_transformers_tokenizer_kwargs may include fix_mistral_regex;
_hf_transformers_model_auth_kwargs never does), bypassing from_model_id
entirely. See model_api.py's own docstrings for the full reasoning, and
tests/test_model_api_tokenizer_fix.py for coverage. This is a real
model-loading correctness fix (tokenizer regex + auth), not a prompt/RAG/
Judge/Defender/pipeline-logic change - re-verified via fresh byte-diff that
all other 15 files in this manifest remain untouched.

final_cn_agent.py's and pipeline.py's hashes WERE deliberately updated
(2026-08-11, v22kg Knowledge Graph work) - the second and third intentional
exceptions. Motivation: the v22kg final-cn-style (see prompts.py section 6b,
counter_narrative_kg.py) builds a lightweight, instance-specific Counter-
Narrative Knowledge Graph and needs it stored in the trace JSONL (not just
used inside the prompt text) - but pipeline.py's trace-building code only
ever cherry-picks specific keys out of final_cn_agent.py's return dict
(counter_narrative/explanation/safety_flags), never the whole dict, so a new
trace key cannot reach the trace from prompts.py alone. The changes are
narrowly scoped and additive-only for every OTHER style:
  - final_cn_agent.py: FinalCNAgent.run() gained an optional
    `case_analysis=None` kwarg (threaded through to
    prompts.build_final_cn_prompt's own new optional `case_analysis=`
    kwarg, used only by v22kg to populate its target_group node) and, only
    when prompts.FINAL_CN_STYLE == "v22kg", attaches two extra keys
    (cn_knowledge_graph, kg_consistency) to its return dict. For every other
    style this branch never runs - the return dict is byte-for-byte the
    same shape as before.
  - pipeline.py: passes case_analysis=case_analysis into
    self.final_cn_agent.run() (new call-site kwarg only), and copies
    cn_knowledge_graph/kg_consistency (plus derived forbidden_claim_nodes/
    required_rebuttal_nodes/approved_factual_evidence_nodes/
    approved_cultural_context_nodes) into the trace dict ONLY when those
    keys are present in `final` - a no-op for every non-v22kg style/older
    trace format.
Re-verified via fresh byte-diff that all other 14 files in this manifest
remain untouched.

final_cn_agent.py's hash was updated AGAIN (2026-08-11, v22kg1 work) - the
fourth intentional exception, pipeline.py unaffected this time. Motivation:
v22kg1 (see prompts.py section 6c, counter_narrative_kg.py's
validate_kg_consistency_full) is a new final-cn-style that reuses v22kg's KG
builder but checks kg_consistency against the actual generated
counter_narrative/explanation too (not just judge_plan), because the real
v22kg Tamil rerun showed unsupported claims appearing only in the model's
own output, which a plan-only validator can never catch. The change is:
FinalCNAgent.run()'s existing `if prompts.FINAL_CN_STYLE == "v22kg":` branch
became `if prompts.FINAL_CN_STYLE in ("v22kg", "v22kg1"):`, with an inner
branch selecting validate_kg_consistency() (unchanged, v22kg only) vs. the
new validate_kg_consistency_full() (v22kg1 only). For every style other than
v22kg1, behavior and output are byte-for-byte identical to before this
change - v22kg's own behavior/trace shape is completely unaffected.
pipeline.py needed NO further edit: its trace-copying is already keyed on
key-presence in `final`, not on style name, so it picks up v22kg1's
cn_knowledge_graph/kg_consistency automatically. Re-verified via fresh
byte-diff that all other 15 files in this manifest remain untouched.

final_cn_agent.py's hash was updated a THIRD time (2026-08-11, v22kg2 work)
- the fifth intentional exception overall, pipeline.py still unaffected.
Motivation: v22kg2 (see prompts.py section 6d, counter_narrative_kg.py's
validate_kg_consistency_v2/has_supported_claim_family) adds full
multilingual (en/ta/eu/es/it) evidence-conditional trigger coverage after
the real v22kg1 Basque rerun showed a science-claim leak (EU125) and an
orientation-immutability leak (EU130) that v22kg1's Tamil/English-only
trigger list had no way to catch. The change is:
`if prompts.FINAL_CN_STYLE in ("v22kg", "v22kg1"):` became
`in ("v22kg", "v22kg1", "v22kg2"):`, with a third inner branch selecting
validate_kg_consistency_v2() (v22kg2 only) alongside the unchanged
validate_kg_consistency()/validate_kg_consistency_full() (v22kg/v22kg1).
For every style other than v22kg2, behavior and output are byte-for-byte
identical to before this change. pipeline.py needed NO further edit (same
key-presence-based trace copying as before). Re-verified via fresh byte-diff
that all other 15 files in this manifest remain untouched.

final_cn_agent.py's hash was updated a FOURTH time (2026-08-11, v22kg3 work)
- the sixth intentional exception overall, pipeline.py still unaffected.
Motivation: v22kg3 (see prompts.py section 6e, counter_narrative_kg.py's
validate_kg_consistency_v3) fixes a real false positive found in the v22kg2
Basque rerun - Basque "dio" (an ordinary auxiliary verb) was being matched
against the Italian religion trigger "dio"=God, because v22kg2's trigger
scan checked every language's phrases at once regardless of the case's own
language. v22kg3 scans per-language instead, and adds a negation guard
("without science/research..." no longer counts as an unsupported claim).
The change is: `if prompts.FINAL_CN_STYLE in ("v22kg", "v22kg1", "v22kg2"):`
became `in ("v22kg", "v22kg1", "v22kg2", "v22kg3"):`, with a fourth inner
branch selecting validate_kg_consistency_v3() (v22kg3 only) alongside the
unchanged validate_kg_consistency()/validate_kg_consistency_full()/
validate_kg_consistency_v2() (v22kg/v22kg1/v22kg2). For every style other
than v22kg3, behavior and output are byte-for-byte identical to before this
change. pipeline.py needed NO further edit. Re-verified via fresh byte-diff
that all other 15 files in this manifest remain untouched.

final_cn_agent.py's hash was updated a FIFTH time (2026-08-11, v22kg4 work)
- the seventh intentional exception overall, pipeline.py still unaffected.
Motivation: v22kg4 is a tiny, prompt-only Basque fluency fallback on top of
v22kg3 - Basque v22kg3 passed the structural KG gates but still produced
awkward/complex Basque wording in real review. Per explicit instruction, KG
architecture, the validator, RAG, Judge, Defender, tokenizer, model, and
evaluation are all untouched; v22kg4 deliberately reuses v22kg3's own
validate_kg_consistency_v3() rather than adding a new validator version.
The change is: `if prompts.FINAL_CN_STYLE in ("v22kg", "v22kg1", "v22kg2",
"v22kg3"):` became `in (..., "v22kg3", "v22kg4"):`, and the inner branch
that selects validate_kg_consistency_v3() now matches
`prompts.FINAL_CN_STYLE in ("v22kg3", "v22kg4")` instead of only "v22kg3" -
both styles share the identical validator call. For every style other than
v22kg4, behavior and output are byte-for-byte identical to before this
change. pipeline.py needed NO further edit. Re-verified via fresh byte-diff
that all other 15 files in this manifest remain untouched.

final_cn_agent.py's hash was updated a SIXTH time (2026-08-11, v22kg5 work)
- the eighth intentional exception overall, pipeline.py still unaffected.
Motivation: v22kg5 is the final, concise multilingual style - Spanish/
Italian/English v22kg4 outputs were safe and fluent but too verbose (3-5
sentences); v22kg5 tightens them to 1-2 sentences and adds English-specific
guardrails (no unsupported biology/brain/hormone framing, no "choose to
love" wording, no unnecessary broadening from sexual orientation to gender
identity). Tamil (v22kg1) and Basque (v22kg4) are explicitly NOT
re-iterated - their prompt text is duplicated unchanged. The validator gets
one correction (validate_kg_consistency_v4, v22kg5 only - validate_kg_
consistency_v3 untouched, still serving v22kg3/v22kg4): a negation guard
for "avoid(s)/avoiding unsupported X" wording, since v22kg5's more explicit
es/it/en explanations sometimes describe method by saying what was AVOIDED.
The change is: `if prompts.FINAL_CN_STYLE in ("v22kg", "v22kg1", "v22kg2",
"v22kg3", "v22kg4"):` became `in (..., "v22kg4", "v22kg5"):`, with a new
first-checked branch `prompts.FINAL_CN_STYLE == "v22kg5"` selecting
validate_kg_consistency_v4(). For every style other than v22kg5, behavior
and output are byte-for-byte identical to before this change. pipeline.py
needed NO further edit. Re-verified via fresh byte-diff that all other 15
files in this manifest remain untouched.

final_cn_agent.py's hash was updated a SEVENTH time (2026-08-11, v22kg6
work) - the ninth intentional exception overall, pipeline.py still
unaffected. Motivation: v22kg6 is a tiny, prompt-only English cleanup on
top of v22kg5 (Spanish/Italian already passed sanity validation and were
explicitly out of scope; Tamil/Basque explicitly not re-iterated). Fixes
three real v22kg5 English issues: "choose to love" wording (can imply
orientation is a choice - now hard-banned with named variants), mental-
disorder/science facts used as a default rebuttal for claim types that
aren't actually disease/pathology framing, and explanations using science-
coded words ("pseudoscientific assertion") that could trip kg_consistency
even when the counter_narrative itself was clean. Per explicit instruction,
validate_kg_consistency_v4 is reused unchanged (confirmed sufficient before
implementing - already doesn't flag "avoids unsupported scientific claims"
or "without using science/legal/religious/country claims", already fixes
the Basque "dio" false positive, still flags real unsupported claims) - no
new validator was written. The change is:
`if prompts.FINAL_CN_STYLE in ("v22kg", "v22kg1", "v22kg2", "v22kg3",
"v22kg4", "v22kg5"):` became `in (..., "v22kg5", "v22kg6"):`, and the inner
branch that selects validate_kg_consistency_v4() now matches
`prompts.FINAL_CN_STYLE in ("v22kg5", "v22kg6")` instead of only "v22kg5" -
both styles share the identical validator call. For every style other than
v22kg6, behavior and output are byte-for-byte identical to before this
change. pipeline.py needed NO further edit. Re-verified via fresh byte-diff
that all other 15 files in this manifest remain untouched.

final_cn_agent.py's hash was updated an EIGHTH time, and pipeline.py's hash
was updated a SECOND time (2026-08-11, v22kg7 work) - the tenth and
eleventh intentional exceptions overall. Motivation: v22kg7 = v22kg6 + a
small, deterministic, no-LLM-call KG safety fallback (English only) -
fixing 4 real v22kg6 English findings: cultural-only brain/biology/hormone
wording still leaked through (v22kg6's validator never had those triggers);
a child/abortion case used mental-disorder evidence instead of a direct
child-dignity rebuttal; unsupported "not supported by science" wording
appeared; and a genuinely relevant mental-disorder rebuttal (hate comment
said "insane", which the shared claim_type classifier doesn't recognize as
disease_or_pathology) was wrongly flagged. New validate_kg_consistency_v5
(counter_narrative_kg.py) fixes both root causes (broader English science
triggers; a disease-mismatch check that also scans the KG's own
harmful_claim/accusation node text, not just claim_type) without touching
validate_kg_consistency_v4 (still serving v22kg5/v22kg6 unchanged) or the
shared claim_type classifier used by every style's KG. The v22kg7 prompt
itself is byte-identical to v22kg6's (confirmed by test) - dispatched via
the same function, no new prompt-building code at all.
final_cn_agent.py's change: a new `if prompts.FINAL_CN_STYLE == "v22kg7":`
branch calls counter_narrative_kg.apply_kg_safety_fallback(), which may
REPLACE counter_narrative/explanation (English only, deterministic,
claim-type-keyword-selected, never a second model call) and always attaches
a new `kg_safety_fallback` trace key. pipeline.py's change: one new
presence-gated block (`if "kg_safety_fallback" in final: trace[...] = ...`),
exactly mirroring the existing cn_knowledge_graph/kg_consistency pattern -
a no-op for every style except v22kg7. For every style other than v22kg7,
behavior and output are byte-for-byte identical to before this change.
Re-verified via fresh byte-diff that all other 14 files in this manifest
remain untouched.

config.py's hash was updated (2026-08-13) - the twelfth intentional
exception overall, and config.py's first. Motivation: not a generation-path
change at all - a dataset-preparation-time fix, added one new constant,
`ML_MTCONAN_TEST_HOLDOUT_FRACTION = 0.2`. Verified directly against the real
LanD-FBK/ML_MTCONAN_KN Hub data that its official test split ships with
every reference counter-narrative (KN_CN) empty, across all 4 languages
(en/eu/es/it) - a genuine leaderboard-style held-out test set, not a loading
bug (train: 0/1584 empty; validation: 0/400 empty; test: 400/400 empty).
Per explicit supervisor direction, dataset_loader.py's new
carve_test_from_train() now carves a fresh, ground-truth-bearing test split
out of the official TRAIN split instead, using this fraction - the official
validation split is left untouched. This constant is read only by
dataset_loader.py, at dataset-load time, never during generation - RAG
retrieval, Judge, Defender, prompts, and model config are all unaffected.
Re-verified via fresh byte-diff that all other 15 files in this manifest
remain untouched.

pipeline.py's hash was updated a THIRD time, config.py's hash was updated a
SECOND time, final_cn_agent.py's hash was updated a NINTH time, and
persona_generator_agent.py's hash was updated for the FIRST time (2026-08-14,
ablation-study work) - the thirteenth through sixteenth intentional
exceptions overall. Motivation: added CLI-driven ablation support
(main.py's --ablation full/no_kg/no_debate/no_persona, --run_all_ablations)
so each of v22kg7's three main contribution modules (Counter-Narrative
Knowledge Graph, multi-agent debate, instance-specific persona selection)
can be individually disabled and measured against the full system - NOT a
new generation behaviour, only a togglable ablation of existing v22kg7
behaviour, off by default. config.py gained ABLATION_MODES/ablation_mode/
set_ablation_mode()/use_kg()/use_debate()/use_instance_personas()/
use_kg_validation()/use_kg_fallback() - all new, additive-only; "full" (the
hard default, unchanged) makes every use_*() function return True.
persona_generator_agent.py gained one new function, generic_persona_
selection() (a fixed, non-instance-specific persona pair, no LLM call) -
PersonaGeneratorAgent.run() itself is untouched. pipeline.py's generate()
now checks config.use_instance_personas()/config.use_debate() before calling
the real persona generator / fast_track/deep_dive_track (calling
generic_persona_selection() / leaving debate_rounds=[] instead when
ablation disables them), and the trace gains 5 new always-present keys
(ablation_mode, debate_outputs, debate_disabled, personas, persona_
selection_disabled) - for config.ablation_mode == "full" every one of these
branches takes the exact same path as before, so behaviour and output are
byte-for-byte identical to pre-ablation v22kg7. final_cn_agent.py's existing
`if prompts.FINAL_CN_STYLE in (v22kg family):` branch gained a sibling
`if ... and not config.use_kg():` branch checked first, which attaches
explicit disabled-stub cn_knowledge_graph/kg_consistency/kg_safety_fallback
dict values instead of building/validating/falling-back through the KG -
counter_narrative/explanation generation itself (the call to
build_final_cn_prompt/client.generate above this block) is completely
unaffected, since the KG was never an input to generation in the first
place (only ever a post-generation validation/fallback artifact - see the
v22kg exception above). For config.use_kg() == True (i.e. every existing
call site, and --ablation full), this new branch is never taken and the
pre-existing elif branch runs unchanged. Re-verified via fresh byte-diff
that all other 12 files in this manifest remain untouched, and via
tests/test_ablation_modes.py that --ablation full reproduces the exact same
trace shape/KG behaviour as before this change.

CORRECTION to the paragraph immediately above: the claim "the KG was never
an input to generation in the first place" was WRONG, and is retracted
here rather than silently edited, per this file's own stated purpose.
prompts.py's `_build_final_cn_prompt_v22kg6()` (used by both v22kg6 and
v22kg7, per prompts.py section 6b/6h) DOES call build_counter_narrative_kg()
and inject a "COUNTER-NARRATIVE KNOWLEDGE GRAPH (PRIMARY GROUNDING
CONTRACT...)" block directly into the prompt text the model actually
generates from - it is not solely a post-generation artifact. This means
the previous --ablation no_kg implementation only ever disabled the
post-generation validation/fallback layer while still generating with the
full KG-grounded prompt underneath - not a true ablation of the KG's
contribution to generation itself. pipeline.py's hash was updated a FOURTH
time, config.py's hash was updated a THIRD time, final_cn_agent.py's hash
was updated a TENTH time, and persona_generator_agent.py's hash was
updated a SECOND time (2026-08-14) to fix this - the seventeenth through
twentieth intentional exceptions overall. prompts.py (not in this manifest,
see its own note above) gained a `use_kg: bool = True` parameter on
build_final_cn_prompt()/_build_final_cn_prompt_v22kg6(): when True (the
default - every existing call site, and --ablation full), the KG is built
and injected exactly as before, producing a byte-identical prompt (verified
by diff of the reconstructed prompt-building code, and by
test_ablation_modes.py::test_full_mode_keeps_kg_prompt_and_trace asserting
the literal "COUNTER-NARRATIVE KNOWLEDGE GRAPH" string is present in the
prompt actually sent to the model); when False (--ablation no_kg only), KG
construction is skipped, a lightweight non-KG claim-type heuristic
(_infer_claim_type_without_kg) substitutes for it, explicit KG wording is
stripped from the reused v22b6 base prompt, and a "NO-GRAPH ABLATION MODE"
instruction block is substituted for the KG grounding block - verified by
test_ablation_modes.py::test_no_kg_removes_kg_from_prompt_and_trace
asserting "COUNTER-NARRATIVE KNOWLEDGE GRAPH" is ABSENT and "NO-GRAPH
ABLATION MODE" is present in the actual prompt sent to the model.
final_cn_agent.py now passes use_kg=config.use_kg() into
build_final_cn_prompt() (previously it never passed this argument at all),
and its post-generation disabled-stub branch is unchanged in effect, with
kg_safety_fallback's stub now only attached for FINAL_CN_STYLE == "v22kg7"
specifically (v22kg1-v22kg6 never produced that key even in "full" mode, so
the stub should not appear for them either under no_kg - a minor schema
tightening). config.py gained ABLATION_BATCH_MODES = ["no_kg", "no_debate",
"no_persona"], used by main.py's --run_all_ablations so it does not re-run
"full" (a deliberate scope change from the original --run_all_ablations
behaviour, not a bug fix - "full" is still available via --ablation full
individually). persona_generator_agent.py's GENERIC_PERSONAS gained
richer expertise/cultural_relevance/strategy fields (cosmetic, no behaviour
change to the no_persona ablation). tests/test_ablation_modes.py was
replaced with a version that captures the actual prompt text sent to the
mocked LLM client and asserts on its content directly, which is what
caught this bug. Re-verified via fresh byte-diff that all other 11 files in
this manifest remain untouched, and via the full test suite (511/511
passing) that no other generation behaviour changed.

persona_generator_agent.py's hash was updated a THIRD time (2026-08-14) -
the twenty-first intentional exception overall. Motivation: not a
correctness fix - the no_persona ablation's fixed GENERIC_PERSONAS/
generic_persona_selection() pair was renamed from abstract role labels
("General Safety Reviewer", "Evidence-Aware Responder", "Respectful
Communication Reviewer") to real-world professional fields ("Legal
Analyst" for prosecutor, "Clinical Psychologist" for defender, "Human
Rights Advocate" as an unselected third candidate), so the no_persona
ablation isolates exactly one variable against "full" (instance-specific
adaptation) instead of also varying "expert-sounding vs. generic-sounding"
persona framing - the real, instance-specific persona generator already
produces professional-title personas (see this file's own module
docstring, e.g. "forensic linguist"), so a fair fixed-baseline comparison
needs the same kind of framing, just non-adaptive. PersonaGeneratorAgent.run()
itself, _fallback_selection(), and every other file are untouched.
tests/test_ablation_modes.py's persona-name assertions were updated to
match. Re-verified via fresh byte-diff that all other 15 files in this
manifest remain untouched, and via the full test suite (511/511 passing).

persona_generator_agent.py's hash was updated a FOURTH time (2026-08-14) -
the twenty-second intentional exception overall. Motivation: the previous
fixed pair (Legal Analyst / Clinical Psychologist) was flagged as a
methodological confound - a domain-flavored fixed defender persona (e.g.
"Clinical Psychologist") risks systematically pulling generation toward
that domain's register (medical/mental-health framing) regardless of the
actual claim type, which is exactly the kind of unsupported-claim risk the
KG is meant to control for. That would make no_persona's result reflect
"domain bias from the fixed persona" as well as "lack of instance-specific
adaptation", confounding the ablation. GENERIC_PERSONAS/generic_persona_
selection() changed to claim-type-agnostic professional roles instead:
prosecutor "Critical Discourse Analyst" (examines framing/logic, not tied
to any one claim register), defender "Human Rights Advocate" (dignity/
rights-based rebuttal, applicable to every claim type this project
handles), with "Evidence Reviewer" as an unselected third candidate.
PersonaGeneratorAgent.run() and every other file are untouched. tests/
test_ablation_modes.py's persona-name assertions were updated to match.
Re-verified via fresh byte-diff that all other 15 files in this manifest
remain untouched, and via the full test suite (511/511 passing).
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_NEW_ARCH_ROOT = Path(__file__).resolve().parent.parent

# Computed once, immediately after confirming (via a fresh extraction of
# colab_transfer_v20.zip and `diff -q`) that every one of these files was
# byte-identical to its v20 counterpart.
_EXPECTED_HASHES = {
    "pipeline.py": "8a1671010a2b17bed3cdfcd9718884f66b06e42158d658df0dab30173ec35014",
    "schemas.py": "306ccad849e79a7a025780dc5daa62a34369cb1a1e13339948b6ca3e7c01d31e",
    "defender_agent.py": "c2023712353a18798d76b35dd80de0c6cebadbdb1427d550f266329e5df632d0",
    "judge_critic_agent.py": "6030ac04d37b5a11eacd3e42824d302bc79ca1d9c79c673b0833e4618b64550a",
    "model_api.py": "9e57507292279d2f5baf8ac902924e7cb1177b24e42ee3b1e13c9d825719d51a",
    "config.py": "698c4357ea21fdd102bc7e9e88806057074f22f6ce3fd68176195e3aae49ed8c",
    "final_cn_agent.py": "7330a6b206c64343513c44f65c7c2d1318805d6e4eb37373c5e68a1d87124d4d",
    "prosecutor_agent.py": "cead0f0f4f6e4cbe9d39c3e56d6e62867faca81bdf20c24a29aac8aeebe14d46",
    "persona_generator_agent.py": "c8aba2ab582310aae68b701867e80cc97fcacb68c580fa663974222ad9339f97",
    "case_analysis_agent.py": "1be21fbfd05effc27cb018c4dda8689c743920bc0314c4e3b32e7c704e29f4f4",
    "deep_dive_track.py": "a95c1ca95e320ffe9e5334c4e7c7dbb1c576bd30a3f8aa8f41632e3c5c7ecaff",
    "fast_track.py": "818acac50a70ff14abc3c1a8b2cd1fd23809d4fb540451469a91d8d74dfe1b1c",
    "router.py": "e71d6f937caf47ea1cdb717b17974b287b0d22f2d0b1b67aed2f305cf278bd90",
    "safety.py": "df6b413c04bbed34338adc53580c1b836d751bc1763255bf96c1955fa8050799",
    "evidence_verifier.py": "5b3bffed0f3fc898296289c67ef174c98994d5eb33407d0d3cee80bbc5828181",
    "app.py": "4683ef3aac192e7f8f22bc37a202e0d09104cacfa28bac4e6bbd91e56456b78d",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generation_path_files_unchanged_from_v20():
    mismatched = []
    for filename, expected in _EXPECTED_HASHES.items():
        actual = _sha256(_NEW_ARCH_ROOT / filename)
        if actual != expected:
            mismatched.append(filename)
    assert not mismatched, (
        f"Generation-path file(s) changed since v20: {mismatched}. If this is an "
        f"intentional generation change (e.g. Mistral prompt work), update "
        f"_EXPECTED_HASHES here deliberately and say so - do not silently accept."
    )


def test_manifest_covers_every_locked_file():
    """Guards against silently shrinking the manifest itself."""
    assert len(_EXPECTED_HASHES) == 16


if __name__ == "__main__":
    test_generation_path_files_unchanged_from_v20()
    test_manifest_covers_every_locked_file()
    print("test_generation_path_frozen.py: ALL PASSED")
