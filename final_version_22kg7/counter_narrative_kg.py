"""
Counter-Narrative Knowledge Graph (v22kg) - a lightweight, INSTANCE-SPECIFIC
grounding layer: JSON nodes + typed edges, built once per case from data the
pipeline already produces (CaseAnalysis, JudgePlan). Deliberately NOT an
ontology, NOT RDF/OWL, NOT a graph database - a plain Python dict, built by
pure functions, no LLM call (per the explicit instruction: "Do not ask the
LLM to invent a KG from scratch unless absolutely necessary" - everything
here is derivable from structured data the pipeline already has).

Thesis framing (verbatim, for the write-up): "The system constructs a
lightweight instance-specific Counter-Narrative Knowledge Graph, represented
as JSON nodes and typed edges, to structure the target group, harmful
claim, accusation type, approved evidence, cultural framing, required
rebuttal, and forbidden unsupported claims before counter-narrative
generation."

Core principle carried over from v22b1-v22b6 (unchanged, just made
structural here instead of prose-only): cultural grounding guides tone/
framing ONLY. Factual/legal/scientific/country-specific/religious claims
must come only from approved_evidence items with source_type EXACTLY
"fact" or EXACTLY "web" - never from "cultural"/"culturale"/missing/
unknown source_type, and never from cultural_guidance or
final_response_plan alone, no matter how confidently those state a claim.

Only used by v22kg's final-CN prompt (prompts.py) and, for trace storage
only, final_cn_agent.py/pipeline.py - see those files' own comments for
exactly how this stays isolated from every other final-cn-style.
"""
import hashlib
import re

_KG_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Node/edge type vocabularies - kept exactly as specified, no ontology/RDF.
# ---------------------------------------------------------------------------
NODE_TYPES = {
    "target_group", "harmful_claim", "protected_attribute", "accusation",
    "claim_type", "approved_factual_evidence", "approved_cultural_context",
    "required_rebuttal", "forbidden_claim", "style_constraint",
    "safety_constraint", "final_response_plan_claim",
}

EDGE_TYPES = {
    "targets", "attacks_attribute", "accuses", "has_claim_type",
    "supported_by", "culturally_framed_by", "requires_rebuttal",
    "forbids_claim", "guides_style", "should_not_strengthen_to",
    "plan_claim_must_be_checked_against", "violates_constraint",
}

_FACT_SOURCE_TYPES = {"fact", "web"}  # exact match only - see module docstring


# ---------------------------------------------------------------------------
# Claim-type classification - lightweight, keyword-based (deliberately NOT
# an LLM call), multilingual (en/ta/eu/es/it substring triggers, checked
# case-insensitively against the raw hate comment + the Judge's
# core_claim_to_counter, since either may be in the target language).
# Falls back to "generic_dehumanization" when nothing matches - a safe,
# always-applicable rebuttal, never left blank.
# ---------------------------------------------------------------------------
_CLAIM_TYPE_KEYWORDS = {
    "crime_generalization": [
        "crime", "criminal", "predator", "pedophile", "pedo", "groomer",
        "குற்ற", "குற்றவாளி",  # ta: crime/criminal
        "delito", "criminal", "delincuen",  # es
        "delitto", "criminale", "criminal",  # it
        "krimen", "delitu",  # eu
    ],
    "silencing": [
        "silence", "silenced", "suppress", "should not speak", "ban them",
        "அடக்க", "தடுக்க", "அழிக்க",  # ta: suppress/block/destroy (speech)
        "silenciar", "callar", "prohibir hablar",  # es
        "zittire", "silenziare", "sopprimere",  # it
        "isilarazi", "debekatu",  # eu
    ],
    "curse_or_religion_framing": [
        "curse", "cursed", "god", "sin", "devil", "demon",
        "சாபம்", "கடவுள்",  # ta: curse/god
        "maldicion", "maldito", "dios", "pecado",  # es
        "maledizione", "maledetto", "dio", "peccato",  # it
        "madarikazio", "jainko",  # eu
    ],
    "disease_or_pathology": [
        "disease", "disorder", "sick", "illness", "cure", "mental illness",
        "நோய்", "மனநோய்",  # ta: disease/mental illness
        "enfermedad", "enfermo", "trastorno", "curar",  # es
        "malattia", "malato", "disturbo", "curare",  # it
        "gaixotasun", "gaixo",  # eu
    ],
    "identity_mockery": [
        "mock", "mocking", "ridicule", "laugh at", "joke about",
        "கேலி", "கிண்டல்",  # ta: mockery/ridicule
        "burla", "ridiculizar", "burlarse",  # es
        "burla", "deridere", "ridicolizzare",  # it
        "trufa", "iseka",  # eu
    ],
    "slur_or_name_calling": [
        "faggot", "maricon", "frocio", "slur",
        "பெயர்",  # ta: name (as in name-calling) - weak signal, low priority
        "insulto",  # es
        "insulto",  # it
    ],
}

# Checked in this fixed order so a comment matching multiple categories
# (common - hate speech often overlaps) gets a stable, predictable
# classification rather than depending on dict iteration order.
_CLAIM_TYPE_PRIORITY = [
    "curse_or_religion_framing", "crime_generalization", "disease_or_pathology",
    "silencing", "slur_or_name_calling", "identity_mockery",
]

_REQUIRED_REBUTTAL_BY_CLAIM_TYPE = {
    "crime_generalization": "Do not blame a whole group for crimes.",
    "silencing": "Silencing someone because of identity is unfair.",
    "curse_or_religion_framing": "Calling someone's identity a curse is harmful.",
    "disease_or_pathology": "LGBTQ+ identity is not a disease or disorder.",
    "identity_mockery": "Mocking someone's identity is wrong.",
    "slur_or_name_calling": "Insulting someone's identity increases harm and prejudice.",
    "generic_dehumanization": "Everyone deserves dignity and equal respect.",
}

_HATE_CATEGORY_TO_ATTRIBUTE = {
    "homophobia": "sexual orientation",
    "transphobia": "gender identity",
    "mixed": "sexual orientation and gender identity",
}


def _classify_claim_type(hate_comment: str, core_claim: str) -> str:
    haystack = f"{hate_comment or ''} {core_claim or ''}".lower()
    for claim_type in _CLAIM_TYPE_PRIORITY:
        for kw in _CLAIM_TYPE_KEYWORDS[claim_type]:
            if kw in haystack:
                return claim_type
    return "generic_dehumanization"


# ---------------------------------------------------------------------------
# Forbidden-claim detection - lightweight keyword gating, per category. A
# category is "supported" (forbidden node NOT added / added in a narrower
# form) only if a source_type EXACTLY "fact" or "web" item's own text
# contains a relevant trigger. Deliberately conservative: cultural-only
# evidence NEVER lifts a forbidden_claim node, matching the exact-scope
# principle carried over from v22b2-v22b6.
# ---------------------------------------------------------------------------
_COUNTRY_REGION_NAMES = [
    "italy", "italia", "europe", "europa", "india", "spain", "españa",
    "basque country", "euskal herria", "tamil nadu",
]
_HEDGE_PHRASES = ["many countries", "molti paesi", "muchos países", "many nations"]
_SCIENCE_RESEARCH_KEYWORDS = [
    "science", "scientific", "research", "study", "studies", "psycholog",
    "medical", "clinical", "data show", "statistic",
]
_LEGAL_KEYWORDS = [
    "law", "legal", "constitution", "court", "legislation", "statute",
    "legally", "human rights act",
]
_RELIGIOUS_INSTITUTIONAL_KEYWORDS = [
    "scripture", "doctrine", "official teaching", "religious law states",
    "canon law", "official position of",
]
_VALUES_LEVEL_ONLY_KEYWORDS = [
    "empathy", "fairness", "compassion", "respect", "spiritual value", "kindness",
]
_STRENGTHENING_TRIGGERS = [
    # (substring to look for in an evidence/cultural passage, forbidden node key
    # it must not be strengthened toward)
    ("natural expression of human diversity", "not_a_choice"),
    ("many countries", "unsupported_country_claim"),
    ("molti paesi", "unsupported_country_claim"),
]


def _fact_web_texts(approved_evidence: list) -> list:
    return [str(e.get("passage") or "") for e in approved_evidence
            if isinstance(e, dict) and e.get("source_type") in _FACT_SOURCE_TYPES]


def _any_keyword_in(texts: list, keywords: list) -> bool:
    joined = " ".join(texts).lower()
    return any(kw in joined for kw in keywords)


class _KGBuilder:
    """Internal helper - not part of the public API. Accumulates nodes/edges
    with auto-incrementing IDs so build_counter_narrative_kg() itself reads
    as a flat, linear sequence of "add this node/edge" calls."""

    def __init__(self):
        self.nodes = []
        self.edges = []
        self._n = 0
        self._e = 0

    def add_node(self, node_type: str, text: str, language: str = "", source: str = "",
                 source_type: str = "", evidence_ids: list = None) -> str:
        assert node_type in NODE_TYPES, f"unknown KG node type: {node_type!r}"
        self._n += 1
        nid = f"n{self._n}"
        self.nodes.append({
            "id": nid, "type": node_type, "text": text, "language": language or "",
            "source": source, "source_type": source_type or "",
            "evidence_ids": evidence_ids or [], "confidence": None,
        })
        return nid

    def add_edge(self, from_node: str, to_node: str, edge_type: str, note: str = "") -> str:
        assert edge_type in EDGE_TYPES, f"unknown KG edge type: {edge_type!r}"
        self._e += 1
        eid = f"e{self._e}"
        self.edges.append({"id": eid, "from_node": from_node, "to_node": to_node,
                            "type": edge_type, "note": note})
        return eid


def build_counter_narrative_kg(hate_comment: str, language: str, case_analysis: dict = None,
                                judge_plan: dict = None, approved_evidence: list = None,
                                region_context: str = None) -> dict:
    """Pure, deterministic builder - no LLM call. case_analysis and
    judge_plan are plain dicts (already-.to_dict()'d) or None; both are
    optional and the graph degrades gracefully (a missing target_group
    becomes an "unclear" node rather than a crash) so this works even when
    upstream data is incomplete. approved_evidence defaults to
    judge_plan["approved_evidence"] when not given explicitly."""
    case_analysis = case_analysis or {}
    judge_plan = judge_plan or {}
    approved_evidence = (approved_evidence if approved_evidence is not None
                          else (judge_plan.get("approved_evidence") or []))
    hate_comment = hate_comment or ""
    core_claim = judge_plan.get("core_claim_to_counter") or ""

    kg = _KGBuilder()

    target_group_text = case_analysis.get("target_group") if case_analysis.get("target_group") else None
    target_id = kg.add_node("target_group", target_group_text or "unclear", language=language,
                             source="case_analysis" if target_group_text else "unavailable")

    claim_id = kg.add_node("harmful_claim", core_claim or hate_comment, language=language,
                            source="judge_plan.core_claim_to_counter" if core_claim else "hate_comment")
    kg.add_edge(claim_id, target_id, "targets")

    hate_category = case_analysis.get("hate_category")
    attribute_text = _HATE_CATEGORY_TO_ATTRIBUTE.get(hate_category, "sexual orientation or gender identity")
    attribute_id = kg.add_node("protected_attribute", attribute_text, language=language,
                                source="case_analysis" if hate_category else "inferred")
    kg.add_edge(claim_id, attribute_id, "attacks_attribute")

    claim_type = _classify_claim_type(hate_comment, core_claim)
    accusation_id = kg.add_node("accusation", core_claim or hate_comment, language=language,
                                 source="heuristic_classifier")
    claim_type_id = kg.add_node("claim_type", claim_type, language=language, source="heuristic_classifier")
    kg.add_edge(claim_id, accusation_id, "accuses")
    kg.add_edge(claim_id, claim_type_id, "has_claim_type")

    # Evidence separation - EXACT source_type match only (see module docstring).
    factual_ids, cultural_ids = [], []
    for i, e in enumerate(approved_evidence):
        if not isinstance(e, dict):
            e = {"passage": str(e), "source_type": ""}
        source_type = e.get("source_type")
        passage = str(e.get("passage") or "")
        evidence_id = str(e.get("source_id") or i)
        if source_type in _FACT_SOURCE_TYPES:
            node_type = "approved_factual_evidence"
        else:
            # "cultural", "culturale", missing, unknown, translated, or any
            # other non-exact value - all fold into cultural_context, never
            # factual, per the exact-source_type-match rule.
            node_type = "approved_cultural_context"
        nid = kg.add_node(node_type, passage, language=language, source="approved_evidence",
                           source_type=source_type or "missing", evidence_ids=[evidence_id])
        if node_type == "approved_factual_evidence":
            factual_ids.append(nid)
            kg.add_edge(claim_type_id, nid, "supported_by")
        else:
            cultural_ids.append(nid)
            kg.add_edge(claim_id, nid, "culturally_framed_by")

    for cg_text in (judge_plan.get("cultural_guidance") or []):
        nid = kg.add_node("approved_cultural_context", cg_text, language=language,
                           source="judge_plan.cultural_guidance", source_type="cultural_guidance")
        kg.add_edge(claim_id, nid, "culturally_framed_by")
        cultural_ids.append(nid)

    # Required rebuttal - directly from claim_type, per Task 5.
    rebuttal_text = _REQUIRED_REBUTTAL_BY_CLAIM_TYPE[claim_type]
    rebuttal_id = kg.add_node("required_rebuttal", rebuttal_text, language=language,
                               source="claim_type_rule")
    kg.add_edge(claim_id, rebuttal_id, "requires_rebuttal")
    for fid in factual_ids:
        kg.add_edge(rebuttal_id, fid, "supported_by")

    # Forbidden claims - conservative, keyword-gated per category.
    fact_texts = _fact_web_texts(approved_evidence)
    forbidden_ids = []

    if not _any_keyword_in(fact_texts, _COUNTRY_REGION_NAMES):
        fid = kg.add_node("forbidden_claim", "unsupported_country_claim", language=language,
                           source="grounding_rule")
        forbidden_ids.append(fid)
        kg.add_node("forbidden_claim", "unsupported_region_claim", language=language,
                     source="grounding_rule")
        forbidden_ids.append(kg.nodes[-1]["id"])
        kg.add_node("forbidden_claim", "unsupported_local_culture_claim", language=language,
                     source="grounding_rule")
        forbidden_ids.append(kg.nodes[-1]["id"])
        kg.add_edge(claim_id, fid, "forbids_claim")

    if not _any_keyword_in(fact_texts, _SCIENCE_RESEARCH_KEYWORDS):
        fid = kg.add_node("forbidden_claim", "unsupported_science_claim", language=language,
                           source="grounding_rule")
        forbidden_ids.append(fid)
        kg.add_node("forbidden_claim", "unsupported_research_claim", language=language,
                     source="grounding_rule")
        forbidden_ids.append(kg.nodes[-1]["id"])
        kg.add_node("forbidden_claim", "unsupported_statistics_claim", language=language,
                     source="grounding_rule")
        forbidden_ids.append(kg.nodes[-1]["id"])
        kg.add_edge(claim_id, fid, "forbids_claim")

    if not _any_keyword_in(fact_texts, _LEGAL_KEYWORDS):
        fid = kg.add_node("forbidden_claim", "unsupported_legal_claim", language=language,
                           source="grounding_rule")
        forbidden_ids.append(fid)
        kg.add_edge(claim_id, fid, "forbids_claim")

    # Religious/theological: ALWAYS forbidden unless fact/web evidence has an
    # explicit institutional marker - "empathy/fairness/spiritual values" in
    # cultural evidence does NOT lift this (Task 4's explicit instruction,
    # directly targeting the real V4_67 finding: "religion accepts everyone"
    # is not licensed just because evidence mentions compassion/respect).
    if not _any_keyword_in(fact_texts, _RELIGIOUS_INSTITUTIONAL_KEYWORDS):
        fid = kg.add_node("forbidden_claim", "unsupported_theological_claim", language=language,
                           source="grounding_rule")
        forbidden_ids.append(fid)
        kg.add_node("forbidden_claim", "unsupported_religion_generalization", language=language,
                     source="grounding_rule")
        forbidden_ids.append(kg.nodes[-1]["id"])
        kg.add_edge(claim_id, fid, "forbids_claim")
        all_evidence_texts = fact_texts + [str(e.get("passage") or "") for e in approved_evidence
                                            if isinstance(e, dict)]
        if _any_keyword_in(all_evidence_texts, _VALUES_LEVEL_ONLY_KEYWORDS):
            style_id = kg.add_node("style_constraint",
                                    "faith should encourage compassion and respect (values-level only)",
                                    language=language, source="grounding_rule")
            kg.add_edge(style_id, forbidden_ids[-2], "guides_style")

    # should_not_strengthen_to edges - risky paraphrase-drift pairs found
    # via keyword scan of ALL evidence text (factual + cultural), per Task 4
    # item 5.
    all_texts_with_ids = [(str(e.get("passage") or ""), i) for i, e in enumerate(approved_evidence)
                           if isinstance(e, dict)]
    for node in list(kg.nodes):
        if node["type"] not in ("approved_factual_evidence", "approved_cultural_context"):
            continue
        lowered = node["text"].lower()
        for trigger, forbidden_key in _STRENGTHENING_TRIGGERS:
            if trigger in lowered:
                target_forbidden = next((n["id"] for n in kg.nodes
                                          if n["type"] == "forbidden_claim" and n["text"] == forbidden_key), None)
                if target_forbidden is None:
                    target_forbidden = kg.add_node("forbidden_claim", forbidden_key, language=language,
                                                    source="grounding_rule")
                kg.add_edge(node["id"], target_forbidden, "should_not_strengthen_to",
                            note=f"must not narrow/strengthen beyond: {trigger!r}")

    # safety_guidance -> safety_constraint nodes (informational, not
    # forbidden claims - these come from the Judge's own explicit safety
    # notes, which are a different, already-trusted signal).
    for sg_text in (judge_plan.get("safety_guidance") or []):
        nid = kg.add_node("safety_constraint", sg_text, language=language, source="judge_plan.safety_guidance")
        kg.add_edge(nid, claim_id, "guides_style")

    # final_response_plan -> final_response_plan_claim node, explicitly
    # linked to every forbidden_claim node so a consumer (prompt text or
    # the validator) can see it must be checked, not trusted as evidence.
    final_response_plan = judge_plan.get("final_response_plan") or ""
    if final_response_plan:
        frp_id = kg.add_node("final_response_plan_claim", final_response_plan, language=language,
                              source="judge_plan.final_response_plan")
        for fnid in forbidden_ids:
            kg.add_edge(frp_id, fnid, "plan_claim_must_be_checked_against")

    graph_id = hashlib.sha256(f"{language}:{hate_comment}".encode("utf-8")).hexdigest()[:16]
    kg_summary = {
        "num_nodes": len(kg.nodes), "num_edges": len(kg.edges),
        "claim_type": claim_type, "target_group": target_group_text or "unclear",
        "num_factual_evidence_nodes": len(factual_ids),
        "num_cultural_context_nodes": len(cultural_ids),
        "num_forbidden_claim_nodes": len(forbidden_ids),
        "schema_version": _KG_SCHEMA_VERSION,
    }

    return {
        "graph_id": graph_id, "language": language, "region_context": region_context or "unknown",
        "nodes": kg.nodes, "edges": kg.edges, "kg_summary": kg_summary,
    }


# ---------------------------------------------------------------------------
# Non-invasive consistency validator (Task 7) - checks judge_plan's
# final_response_plan/cultural_guidance text against the KG's
# forbidden_claim nodes. NEVER blocks generation, NEVER rewrites
# JudgePlan - diagnostic only, meant to be logged into the trace so the
# thesis can show the KG catches unsupported plan content even before any
# upstream (Judge/Defender) change.
# ---------------------------------------------------------------------------
_FORBIDDEN_KEY_TRIGGER_PHRASES = {
    "unsupported_country_claim": _COUNTRY_REGION_NAMES,
    "unsupported_region_claim": _COUNTRY_REGION_NAMES,
    "unsupported_local_culture_claim": _COUNTRY_REGION_NAMES,
    "unsupported_science_claim": ["science confirms", "science shows", "scientific evidence",
                                   "அறிவியல் அடிப்படையில்"],
    "unsupported_research_claim": ["research shows", "studies show", "research confirms",
                                    "ஆராய்ச்சி காட்டுகிறது"],
    "unsupported_statistics_claim": ["statistics show", "data shows"],
    "unsupported_legal_claim": ["law protects", "laws protect", "legally guarantee", "law guarantees",
                                 "சட்டம் கூறுகிறது"],
    # Tamil phrases below are drawn directly from the real V4_67 case (see
    # counter_narrative_kg.py's module docstring and test_counter_narrative_kg.py) -
    # the Judge's own final_response_plan text is often in the target language, not English,
    # so an English-only trigger list would never catch the exact failure this validator exists for.
    "unsupported_theological_claim": ["god loves", "god hates", "god does not", "god accepts",
                                       "கடவுளின் அன்பு", "கடவுள் நேசிக்கிறார்", "கடவுள் வெறுக்கவில்லை"],
    "unsupported_religion_generalization": ["religion accepts", "religion has principles",
                                             "all religions", "every religion",
                                             "ஏற்றுக்கொள்ளும் கொள்கைகளை", "மதம் அனைவரையும்",
                                             "அனைத்து மதங்களும்"],
    "not_a_choice": ["not a choice", "not choose", "cannot choose"],
}


def validate_kg_consistency(kg: dict, judge_plan: dict) -> dict:
    """Returns {"final_response_plan_violations": [...],
    "forbidden_claims_triggered": [...], "risk_level": "none"/"low"/"medium"/
    "high"}. Scans judge_plan's final_response_plan + cultural_guidance text
    against every forbidden_claim node's trigger-phrase list. Purely
    additive/diagnostic - the caller decides what (if anything) to do with
    this; nothing here touches JudgePlan or blocks generation."""
    judge_plan = judge_plan or {}
    text_to_check = " ".join([
        str(judge_plan.get("final_response_plan") or ""),
        *[str(x) for x in (judge_plan.get("cultural_guidance") or [])],
    ]).lower()

    forbidden_keys = {n["text"] for n in kg.get("nodes", []) if n["type"] == "forbidden_claim"}
    violations = []
    triggered = []
    for key in forbidden_keys:
        phrases = _FORBIDDEN_KEY_TRIGGER_PHRASES.get(key, [])
        hits = [p for p in phrases if p in text_to_check]
        if hits:
            violations.append({"forbidden_claim": key, "matched_phrases": hits})
            triggered.append(key)

    if not triggered:
        risk_level = "none"
    elif len(triggered) == 1:
        risk_level = "low"
    elif len(triggered) <= 3:
        risk_level = "medium"
    else:
        risk_level = "high"

    return {
        "final_response_plan_violations": violations,
        "forbidden_claims_triggered": triggered,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# v22kg1-only extensions. validate_kg_consistency() above is left completely
# untouched - v22kg keeps using it, unchanged, exactly as before. Everything
# below is new and additive, used only by the v22kg1 final-cn-style.
#
# Motivated by real Tamil v22kg findings: (1) V4_67's counter_narrative still
# generalized about "religions" in NEW wording ("மதங்கள் அன்பு, மரியாதை,
# மற்றும் ஒப்புக்கொள்ளல் ... ஊக்குவிக்கின்றன") that the original, narrower
# trigger-phrase list didn't cover, and the ORIGINAL validate_kg_consistency
# only ever looked at judge_plan text, never at the actual generated output -
# so it could never have caught this even with a broader phrase list. (2)
# V4_64's counter_narrative code-mixed a stray French word ("diversité").
# (3) V4_668's explanation used disease/disorder framing for a claim_type
# that was never about disease. None of these are things a plan-only,
# fixed-phrase-only validator can ever catch.
# ---------------------------------------------------------------------------
_LATIN_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
_CODE_MIX_WHITELIST = {"lgbtq", "lgbtqia"}


def _detect_code_mixing(text: str) -> list:
    """Language-agnostic: finds Latin-script tokens (including accented ones, e.g.
    'diversité') in a string, excluding a small whitelist of entity labels
    (LGBTQ+/LGBTQIA+). Deliberately a general detector rather than a hand-maintained
    list of specific foreign words, since the next code-mixed word won't be one we've
    already seen."""
    if not text:
        return []
    found = []
    for tok in _LATIN_TOKEN_RE.findall(text):
        if tok.lower().rstrip("+") in _CODE_MIX_WHITELIST:
            continue
        found.append(tok)
    return found


_DISEASE_TRIGGERS = ["கோளாறு", "நோய்", "disorder", "disease", "pathology", "medical condition"]
# Claim types where disease/disorder wording is actually appropriate - everything else flags a
# mismatch (see V4_668: identity_mockery explanation wrongly framed as disease/disorder-related).
_DISEASE_ALLOWED_CLAIM_TYPES = {"disease_or_pathology"}

# Broader than _FORBIDDEN_KEY_TRIGGER_PHRASES (kept untouched above, still used by v22kg) - adds
# real-world Tamil rewordings of the same unsupported claims found in the v22kg Tamil rerun.
# Falls back to _FORBIDDEN_KEY_TRIGGER_PHRASES for any key not overridden here.
_FORBIDDEN_KEY_TRIGGER_PHRASES_V2 = {
    "unsupported_theological_claim": _FORBIDDEN_KEY_TRIGGER_PHRASES["unsupported_theological_claim"] + [
        "கடவுள்", "இறைவன்", "ஆன்மிக", "god", "divine",
    ],
    "unsupported_religion_generalization": _FORBIDDEN_KEY_TRIGGER_PHRASES["unsupported_religion_generalization"] + [
        "மதம்", "மதங்கள்", "மத மரபு", "மத மரபுகள்", "மத மரபுகளில்", "உள்ளடக்கிய தன்மை",
        "ஒப்புக்கொள்ளல்", "ஏற்றுக்கொள்ளல்", "அனைவரையும் ஏற்றுக்கொள்ளும்",
        "religion", "religions", "faith tradition", "faith traditions",
        "religious tradition", "religious traditions",
    ],
    "unsupported_science_claim": _FORBIDDEN_KEY_TRIGGER_PHRASES["unsupported_science_claim"] + [
        "அறிவியல் அடிப்படையில்", "அறிவியல் மற்றும் பண்பாடு அடிப்படையில்",
    ],
    "unsupported_research_claim": _FORBIDDEN_KEY_TRIGGER_PHRASES["unsupported_research_claim"] + [
        "ஆய்வுகள்", "புள்ளிவிவரம்",
    ],
}


def validate_kg_consistency_full(kg: dict, judge_plan: dict, counter_narrative: str = "",
                                  explanation: str = "", claim_type: str = None) -> dict:
    """v22kg1-only. Checks THREE things (validate_kg_consistency only ever checked the first):
    final_response_plan/cultural_guidance, the actual generated counter_narrative, and the
    actual generated explanation - each against the broader _FORBIDDEN_KEY_TRIGGER_PHRASES_V2
    list, plus Tamil code-mixing and claim_type/disease-wording mismatch checks.

    Returns {"final_response_plan_violations", "counter_narrative_violations",
    "explanation_violations", "forbidden_claims_triggered", "risk_level"}.

    Diagnostic only, exactly like validate_kg_consistency() - never blocks generation, never
    rewrites JudgePlan, never touches Judge/Defender."""
    judge_plan = judge_plan or {}
    claim_type = claim_type or (kg.get("kg_summary") or {}).get("claim_type")
    forbidden_keys = {n["text"] for n in kg.get("nodes", []) if n["type"] == "forbidden_claim"}

    def _scan(text: str):
        text_l = (text or "").lower()
        violations, triggered = [], []
        for key in forbidden_keys:
            phrases = _FORBIDDEN_KEY_TRIGGER_PHRASES_V2.get(key) or _FORBIDDEN_KEY_TRIGGER_PHRASES.get(key, [])
            hits = [p for p in phrases if p.lower() in text_l]
            if hits:
                violations.append({"forbidden_claim": key, "matched_phrases": hits})
                triggered.append(key)
        return violations, triggered

    plan_text = " ".join([
        str(judge_plan.get("final_response_plan") or ""),
        *[str(x) for x in (judge_plan.get("cultural_guidance") or [])],
    ])
    plan_violations, plan_triggered = _scan(plan_text)
    cn_violations, cn_triggered = _scan(counter_narrative)
    exp_violations, exp_triggered = _scan(explanation)

    language = kg.get("language")
    if language == "ta":
        cn_code_mixing = _detect_code_mixing(counter_narrative)
        exp_code_mixing = _detect_code_mixing(explanation)
        if cn_code_mixing:
            cn_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": cn_code_mixing})
        if exp_code_mixing:
            exp_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": exp_code_mixing})

    disease_allowed = claim_type in _DISEASE_ALLOWED_CLAIM_TYPES
    if not disease_allowed:
        cn_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (counter_narrative or "").lower()]
        exp_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (explanation or "").lower()]
        if cn_disease_hits:
            cn_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": cn_disease_hits})
        if exp_disease_hits:
            exp_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": exp_disease_hits})

    cn_has_issue = bool(cn_violations)
    exp_has_issue = bool(exp_violations)
    plan_has_issue = bool(plan_violations)

    if cn_has_issue or (plan_has_issue and exp_has_issue):
        risk_level = "high"
    elif exp_has_issue:
        risk_level = "medium"
    elif plan_has_issue:
        risk_level = "low"
    else:
        risk_level = "none"

    forbidden_claims_triggered = sorted(set(plan_triggered + cn_triggered + exp_triggered))

    return {
        "final_response_plan_violations": plan_violations,
        "counter_narrative_violations": cn_violations,
        "explanation_violations": exp_violations,
        "forbidden_claims_triggered": forbidden_claims_triggered,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# v22kg2-only extensions. validate_kg_consistency() and validate_kg_consistency_full()
# above are left completely untouched - v22kg and v22kg1 keep using them, unchanged.
#
# Core correction from v22kg1: science/legal/religious/country/cultural TERMS must never be
# blindly banned - a term is only a problem when the corresponding claim is UNSUPPORTED. This
# module already makes that distinction structurally: build_counter_narrative_kg() only ever
# adds an unsupported_* forbidden_claim node when NO approved_factual_evidence (source_type
# EXACTLY "fact"/"web") supports that claim family for this case - so gating a trigger-phrase
# scan on "is this forbidden_claim node actually present in the KG" (not on the trigger phrase
# alone) is already evidence-conditional, not a blind ban. has_supported_claim_family() below
# makes that existing logic an explicit, directly-testable, reusable check rather than only an
# implicit side effect of node presence.
#
# Motivated by the real v22kg1 Basque rerun: EU125's counter_narrative stated "zientziaren eta
# psikologiaren arabera" (according to science and psychology) from evidence that was all
# source_type "cultural" - a real, confirmed leak that validate_kg_consistency_full() (Tamil/
# English trigger phrases only) had no way to catch for Basque wording. EU130 stated, as flat
# fact, "Ez da posible sexualitatea aldatzea" (it's not possible to change sexual orientation) -
# an immutability/conversion claim not covered by any of the KG builder's 4 default categories
# at all, so it needed a 5th, validator-only claim family (orientation_immutability) rather than
# a new KG node type (a new node type would change build_counter_narrative_kg()'s output shape
# for v22kg/v22kg1 too, which must stay unchanged - see this function's own docstring).
# ---------------------------------------------------------------------------
_CLAIM_FAMILY_TO_FORBIDDEN_KEYS = {
    "science_research": ["unsupported_science_claim", "unsupported_research_claim", "unsupported_statistics_claim"],
    "legal": ["unsupported_legal_claim"],
    "religion_theology": ["unsupported_theological_claim", "unsupported_religion_generalization"],
    "country_region": ["unsupported_country_claim", "unsupported_region_claim"],
    "local_culture": ["unsupported_local_culture_claim"],
}

# Evidence phrases that, if present in an approved_factual_evidence node's own text, mean the
# orientation_immutability family IS supported for this case (the one family not derived from a
# KG forbidden_claim node - see module comment above).
_IMMUTABILITY_SUPPORT_TRIGGERS = [
    "cannot be changed", "can not be changed", "is not a choice", "is innate",
    "born this way", "immutable",
]
_IMMUTABILITY_CLAIM_TRIGGERS_MULTI = [
    # Basque - the real EU130 leak, plus close variants
    "ez da posible sexualitatea aldatzea", "ezin da aldatu", "ezin da sendatu",
    # English / Spanish / Italian - same claim family, kept lightweight (not the focus of the
    # real finding, but trivial to cover alongside it)
    "cannot be changed", "impossible to change",
    "no se puede cambiar", "non si può cambiare",
]


def has_supported_claim_family(kg: dict, family: str) -> bool:
    """v22kg2-only. True only when this case's APPROVED FACTUAL/WEB evidence (the
    "approved_factual_evidence" node type - build_counter_narrative_kg() already folds
    source_type EXACTLY "fact"/"web" evidence into this type, see that function's own
    docstring) explicitly supports the given claim family. Cultural-only support, a missing/
    unknown source_type, or vague evidence all return False here, exactly as they do inside
    build_counter_narrative_kg() itself - this function does not re-derive that judgment, it
    just exposes it (for the 4 KG-builder-derived families) or computes it directly against
    approved_factual_evidence text (for "orientation_immutability", the one family that isn't
    one of the KG builder's default categories).

    family: one of "science_research", "legal", "religion_theology", "country_region",
    "local_culture", "orientation_immutability"."""
    if family == "orientation_immutability":
        fact_texts = [n["text"] for n in kg.get("nodes", []) if n["type"] == "approved_factual_evidence"]
        joined = " ".join(fact_texts).lower()
        return any(t in joined for t in _IMMUTABILITY_SUPPORT_TRIGGERS)
    keys = _CLAIM_FAMILY_TO_FORBIDDEN_KEYS.get(family, [])
    present_forbidden = {n["text"] for n in kg.get("nodes", []) if n["type"] == "forbidden_claim"}
    return not any(k in present_forbidden for k in keys)


# Multilingual (en/ta/eu/es/it) trigger phrases per forbidden_claim key - unlike
# _FORBIDDEN_KEY_TRIGGER_PHRASES_V2 (Tamil/English only, kept untouched for v22kg1), these cover
# every language this project supports. Still only ever scanned for a key that's actually
# present as a forbidden_claim node in this case's KG (see validate_kg_consistency_v2 below) -
# i.e. these are triggers for detecting an UNSUPPORTED claim, never a blind ban on the words
# themselves. Per-language phrases are exactly the ones specified for this work, organized by
# which forbidden_claim key each maps to.
_FORBIDDEN_KEY_TRIGGERS_MULTI = {
    "unsupported_science_claim": [
        "science", "scientific", "scientifically", "according to science", "psychology",
        "அறிவியல்", "அறிவியல் அடிப்படையில்", "அறிவியல் மற்றும் பண்பாடு அடிப்படையில்",
        "zientzia", "zientziaren", "zientifiko", "zientifikoki", "psikologia", "psikologiaren",
        "zientziaren eta psikologiaren arabera",
        "ciencia", "científico", "científicamente", "psicología", "según la ciencia",
        "scienza", "scientifico", "scientificamente", "psicologia", "secondo la scienza",
    ],
    "unsupported_research_claim": [
        "research", "studies show", "evidence shows",
        "ஆராய்ச்சி", "ஆய்வுகள்",
        "ikerketa", "ikerketek", "azterketa",
        "estudios", "investigaciones",
        "studi", "ricerche",
    ],
    "unsupported_statistics_claim": [
        "statistics", "data shows",
        "புள்ளிவிவரம்",
        "datuak", "estatistikak",
        "datos", "estadísticas",
        "dati", "statistiche",
    ],
    "unsupported_legal_claim": [
        "law", "legal", "court", "constitution", "rights law", "protected by law",
        "சட்டம்", "நீதிமன்றம்", "அரசியலமைப்பு", "சட்டரீதியாக", "உரிமை சட்டம்",
        "legea", "legez", "auzitegia", "konstituzioa", "legeak babesten du",
        "ley", "tribunal", "constitución", "protegido por la ley",
        "legge", "tribunale", "costituzione", "protetto dalla legge", "la legge garantisce",
    ],
    "unsupported_theological_claim": [
        "god", "god loves", "god hates no one",
        "கடவுள்", "இறைவன்", "ஆன்மிக",
        "jainkoa", "fedea", "espiritual",
        "dios", "dios ama",
        "dio", "dio ama",
    ],
    "unsupported_religion_generalization": [
        "religion", "religions", "religious traditions", "faith traditions", "religion accepts everyone",
        "மதம்", "மதங்கள்", "மத மரபு", "மத மரபுகள்", "மத மரபுகளில்", "உள்ளடக்கிய தன்மை",
        "ஒப்புக்கொள்ளல்", "ஏற்றுக்கொள்ளல்", "அனைவரையும் ஏற்றுக்கொள்ளும்",
        "erlijioa", "erlijioak", "erlijio tradizioak", "onarpena", "onartzen dute",
        "religión", "religiones", "tradiciones religiosas", "la religión acepta",
        "religione", "religioni", "tradizioni religiose", "la religione accetta",
    ],
    "unsupported_country_claim": [
        "in india", "in italy", "in europe", "indian society", "italian culture", "european values",
        "இந்தியா", "இந்தியாவில்", "தமிழ்நாடு", "இந்திய சமூகம்",
        "euskal herria", "europan", "indian", "italian",
        "españa", "en españa", "europa", "en europa", "india", "italia",
        "in italia", "in europa",
    ],
    "unsupported_region_claim": [
        "in india", "in italy", "in europe", "indian society", "italian culture", "european values",
        "இந்தியா", "இந்தியாவில்", "தமிழ்நாடு", "இந்திய சமூகம்",
        "euskal herria", "europan", "indian", "italian",
        "españa", "en españa", "europa", "en europa", "india", "italia",
        "in italia", "in europa",
    ],
    "unsupported_local_culture_claim": [
        "tamil culture", "basque society",
        "தமிழ் கலாச்சாரம்", "உள்ளூர் கலாச்சாரம்",
        "euskal gizartea", "euskal kultura", "tokiko kultura",
        "cultura española", "sociedad española", "cultura italiana", "società italiana",
    ],
}


def validate_kg_consistency_v2(kg: dict, judge_plan: dict, counter_narrative: str = "",
                                explanation: str = "", claim_type: str = None) -> dict:
    """v22kg2-only. Same 3-field shape as validate_kg_consistency_full() (final_response_plan,
    counter_narrative, explanation, plus code-mixing/disease-mismatch), but with full
    multilingual (en/ta/eu/es/it) trigger coverage instead of Tamil/English only, PLUS the
    orientation_immutability family (not derived from a KG node - see has_supported_claim_family
    docstring). Every trigger-phrase check is evidence-conditional: a phrase is only scanned for
    if the case's own KG actually flags that claim family as currently unsupported (or, for
    orientation_immutability, if has_supported_claim_family() says so) - never a blanket ban."""
    judge_plan = judge_plan or {}
    claim_type = claim_type or (kg.get("kg_summary") or {}).get("claim_type")
    present_forbidden = {n["text"] for n in kg.get("nodes", []) if n["type"] == "forbidden_claim"}
    immutability_unsupported = not has_supported_claim_family(kg, "orientation_immutability")

    def _scan(text: str):
        text_l = (text or "").lower()
        violations, triggered = [], []
        for key in present_forbidden:
            phrases = _FORBIDDEN_KEY_TRIGGERS_MULTI.get(key, [])
            hits = [p for p in phrases if p.lower() in text_l]
            if hits:
                violations.append({"forbidden_claim": key, "matched_phrases": hits})
                triggered.append(key)
        if immutability_unsupported:
            hits = [p for p in _IMMUTABILITY_CLAIM_TRIGGERS_MULTI if p.lower() in text_l]
            if hits:
                violations.append({"forbidden_claim": "unsupported_immutability_claim", "matched_phrases": hits})
                triggered.append("unsupported_immutability_claim")
        return violations, triggered

    plan_text = " ".join([
        str(judge_plan.get("final_response_plan") or ""),
        *[str(x) for x in (judge_plan.get("cultural_guidance") or [])],
    ])
    plan_violations, plan_triggered = _scan(plan_text)
    cn_violations, cn_triggered = _scan(counter_narrative)
    exp_violations, exp_triggered = _scan(explanation)

    # Code-mixing detection only makes sense for non-Latin-script languages (Tamil) - applying it
    # to Basque/Spanish/Italian/English would flag every word, since those are Latin-script too.
    language = kg.get("language")
    if language == "ta":
        cn_code_mixing = _detect_code_mixing(counter_narrative)
        exp_code_mixing = _detect_code_mixing(explanation)
        if cn_code_mixing:
            cn_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": cn_code_mixing})
        if exp_code_mixing:
            exp_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": exp_code_mixing})

    disease_allowed = claim_type in _DISEASE_ALLOWED_CLAIM_TYPES
    if not disease_allowed:
        cn_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (counter_narrative or "").lower()]
        exp_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (explanation or "").lower()]
        if cn_disease_hits:
            cn_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": cn_disease_hits})
        if exp_disease_hits:
            exp_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": exp_disease_hits})

    cn_has_issue = bool(cn_violations)
    exp_has_issue = bool(exp_violations)
    plan_has_issue = bool(plan_violations)

    if cn_has_issue or (plan_has_issue and exp_has_issue):
        risk_level = "high"
    elif exp_has_issue:
        risk_level = "medium"
    elif plan_has_issue:
        risk_level = "low"
    else:
        risk_level = "none"

    forbidden_claims_triggered = sorted(set(plan_triggered + cn_triggered + exp_triggered))

    return {
        "final_response_plan_violations": plan_violations,
        "counter_narrative_violations": cn_violations,
        "explanation_violations": exp_violations,
        "forbidden_claims_triggered": forbidden_claims_triggered,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# v22kg3-only extensions. validate_kg_consistency_v2() above is left completely
# untouched - v22kg2 keeps using it, unchanged.
#
# Real false-positive found in the v22kg2 Basque rerun: _FORBIDDEN_KEY_TRIGGERS_MULTI's
# "unsupported_theological_claim" list included bare "dio" (intended as Italian "Dio" = God),
# scanned against EVERY language's text regardless of the case's own language. In Basque, "dio"
# is an ordinary auxiliary-verb form (roughly "he/she says/has"), with zero religious meaning -
# so any Basque sentence using it at all would have been misflagged as an unsupported
# theological claim. Fix: trigger phrases are now organized PER LANGUAGE
# (_FORBIDDEN_KEY_TRIGGERS_BY_LANG) and only scanned using the case's own KG language (kg
# ["language"]) plus English (the project's common evidence/corpus language, so a stray English
# phrase is still caught even in non-English output) - never every language's phrases at once.
# Basque's own theology trigger list deliberately does NOT include "dio" (only "jainkoa"); Italian
# keeps "dio"/"dio ama" as real Italian religion triggers, scanned only when kg["language"]=="it".
# ---------------------------------------------------------------------------
_FORBIDDEN_KEY_TRIGGERS_BY_LANG = {
    "unsupported_science_claim": {
        "en": ["science", "scientific", "scientifically", "according to science", "psychology"],
        "ta": ["அறிவியல்", "அறிவியல் அடிப்படையில்", "அறிவியல் மற்றும் பண்பாடு அடிப்படையில்"],
        "eu": ["zientzia", "zientziaren", "zientifiko", "zientifikoki", "psikologia", "psikologiaren",
               "zientziaren eta psikologiaren arabera"],
        "es": ["ciencia", "científico", "científicamente", "psicología", "según la ciencia"],
        "it": ["scienza", "scientifico", "scientificamente", "psicologia", "secondo la scienza"],
    },
    "unsupported_research_claim": {
        "en": ["research", "studies show", "evidence shows"],
        "ta": ["ஆராய்ச்சி", "ஆய்வுகள்"],
        "eu": ["ikerketa", "ikerketek", "azterketa"],
        "es": ["estudios", "investigaciones"],
        "it": ["studi", "ricerche"],
    },
    "unsupported_statistics_claim": {
        "en": ["statistics", "data shows"],
        "ta": ["புள்ளிவிவரம்"],
        "eu": ["datuak", "estatistikak"],
        "es": ["datos", "estadísticas"],
        "it": ["dati", "statistiche"],
    },
    "unsupported_legal_claim": {
        "en": ["law", "legal", "court", "constitution", "rights law", "protected by law"],
        "ta": ["சட்டம்", "நீதிமன்றம்", "அரசியலமைப்பு", "சட்டரீதியாக", "உரிமை சட்டம்"],
        "eu": ["legea", "legez", "auzitegia", "konstituzioa", "legeak babesten du"],
        "es": ["ley", "tribunal", "constitución", "protegido por la ley"],
        "it": ["legge", "tribunale", "costituzione", "protetto dalla legge", "la legge garantisce"],
    },
    "unsupported_theological_claim": {
        "en": ["god", "god loves", "god hates no one"],
        "ta": ["கடவுள்", "இறைவன்", "ஆன்மிக"],
        "eu": ["jainkoa"],  # deliberately NOT "dio" - see module comment above
        "es": ["dios", "dios ama"],
        "it": ["dio", "dio ama"],
    },
    "unsupported_religion_generalization": {
        "en": ["religion", "religions", "religious traditions", "faith traditions", "religion accepts everyone"],
        "ta": ["மதம்", "மதங்கள்", "மத மரபு", "மத மரபுகள்", "மத மரபுகளில்", "உள்ளடக்கிய தன்மை",
               "ஒப்புக்கொள்ளல்", "ஏற்றுக்கொள்ளல்", "அனைவரையும் ஏற்றுக்கொள்ளும்"],
        "eu": ["erlijioa", "erlijioak", "erlijio tradizioak", "onarpena", "onartzen dute"],
        "es": ["religión", "religiones", "tradiciones religiosas", "la religión acepta"],
        "it": ["religione", "religioni", "tradizioni religiose", "la religione accetta"],
    },
    "unsupported_country_claim": {
        "en": ["in india", "in italy", "in europe", "indian society", "italian culture", "european values"],
        "ta": ["இந்தியா", "இந்தியாவில்", "தமிழ்நாடு", "இந்திய சமூகம்"],
        "eu": ["euskal herria", "europan", "indian", "italian"],
        "es": ["españa", "en españa", "europa", "en europa", "india", "italia"],
        "it": ["in italia", "in europa"],
    },
    "unsupported_region_claim": {
        "en": ["in india", "in italy", "in europe", "indian society", "italian culture", "european values"],
        "ta": ["இந்தியா", "இந்தியாவில்", "தமிழ்நாடு", "இந்திய சமூகம்"],
        "eu": ["euskal herria", "europan", "indian", "italian"],
        "es": ["españa", "en españa", "europa", "en europa", "india", "italia"],
        "it": ["in italia", "in europa"],
    },
    "unsupported_local_culture_claim": {
        "en": ["tamil culture", "basque society"],
        "ta": ["தமிழ் கலாச்சாரம்", "உள்ளூர் கலாச்சாரம்"],
        "eu": ["euskal gizartea", "euskal kultura", "tokiko kultura"],
        "es": ["cultura española", "sociedad española"],
        "it": ["cultura italiana", "società italiana"],
    },
}

# Task 7: negated mentions ("without science/research...", "...zehatzik gabe") must not count as
# an unsupported claim - the text is explicitly saying that basis was NOT used. Scoped per
# sentence-like segment (split on . ! ? ;), not the whole text, and checked without assuming word
# order (Basque negation is often postpositional, e.g. "...gabe" comes AFTER the noun it negates).
_NEGATION_MARKERS = [
    "without", "no science", "no research", "not based on",
    "gabe", "gabeko",
    "sin ", "no según",
    "senza",
    "இல்லாமல்", "இல்லை",
]


def _segment_containing(text_l: str, phrase_l: str) -> str:
    import re as _re
    for segment in _re.split(r"[.!?;]", text_l):
        if phrase_l in segment:
            return segment
    return text_l


def _is_negated_mention(text_l: str, phrase_l: str) -> bool:
    segment = _segment_containing(text_l, phrase_l)
    return any(neg in segment for neg in _NEGATION_MARKERS)


def validate_kg_consistency_v3(kg: dict, judge_plan: dict, counter_narrative: str = "",
                                explanation: str = "", claim_type: str = None) -> dict:
    """v22kg3-only. Same evidence-conditional design as validate_kg_consistency_v2 (has_
    supported_claim_family still gates which claim families are even scanned for), but with two
    corrections found in the real v22kg2 Basque rerun: (1) trigger phrases are now scanned
    PER-LANGUAGE (kg["language"] + English) instead of every language's phrases at once - fixes
    the Basque "dio" (ordinary auxiliary verb) vs Italian "Dio" (God) false positive; (2) a
    negated mention ("without science/research...") is not counted as an unsupported claim."""
    judge_plan = judge_plan or {}
    claim_type = claim_type or (kg.get("kg_summary") or {}).get("claim_type")
    present_forbidden = {n["text"] for n in kg.get("nodes", []) if n["type"] == "forbidden_claim"}
    immutability_unsupported = not has_supported_claim_family(kg, "orientation_immutability")
    language = kg.get("language")
    langs_to_check = {language, "en"} if language else {"en"}

    def _scan(text: str):
        text_l = (text or "").lower()
        violations, triggered = [], []
        for key in present_forbidden:
            per_lang = _FORBIDDEN_KEY_TRIGGERS_BY_LANG.get(key, {})
            phrases = [p for lang in langs_to_check for p in per_lang.get(lang, [])]
            hits = [p for p in phrases if p.lower() in text_l and not _is_negated_mention(text_l, p.lower())]
            if hits:
                violations.append({"forbidden_claim": key, "matched_phrases": hits})
                triggered.append(key)
        if immutability_unsupported:
            hits = [p for p in _IMMUTABILITY_CLAIM_TRIGGERS_MULTI
                    if p.lower() in text_l and not _is_negated_mention(text_l, p.lower())]
            if hits:
                violations.append({"forbidden_claim": "unsupported_immutability_claim", "matched_phrases": hits})
                triggered.append("unsupported_immutability_claim")
        return violations, triggered

    plan_text = " ".join([
        str(judge_plan.get("final_response_plan") or ""),
        *[str(x) for x in (judge_plan.get("cultural_guidance") or [])],
    ])
    plan_violations, plan_triggered = _scan(plan_text)
    cn_violations, cn_triggered = _scan(counter_narrative)
    exp_violations, exp_triggered = _scan(explanation)

    if language == "ta":
        cn_code_mixing = _detect_code_mixing(counter_narrative)
        exp_code_mixing = _detect_code_mixing(explanation)
        if cn_code_mixing:
            cn_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": cn_code_mixing})
        if exp_code_mixing:
            exp_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": exp_code_mixing})

    disease_allowed = claim_type in _DISEASE_ALLOWED_CLAIM_TYPES
    if not disease_allowed:
        cn_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (counter_narrative or "").lower()]
        exp_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (explanation or "").lower()]
        if cn_disease_hits:
            cn_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": cn_disease_hits})
        if exp_disease_hits:
            exp_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": exp_disease_hits})

    cn_has_issue = bool(cn_violations)
    exp_has_issue = bool(exp_violations)
    plan_has_issue = bool(plan_violations)

    if cn_has_issue or (plan_has_issue and exp_has_issue):
        risk_level = "high"
    elif exp_has_issue:
        risk_level = "medium"
    elif plan_has_issue:
        risk_level = "low"
    else:
        risk_level = "none"

    forbidden_claims_triggered = sorted(set(plan_triggered + cn_triggered + exp_triggered))

    return {
        "final_response_plan_violations": plan_violations,
        "counter_narrative_violations": cn_violations,
        "explanation_violations": exp_violations,
        "forbidden_claims_triggered": forbidden_claims_triggered,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# v22kg5-only extensions. validate_kg_consistency_v3() above is left completely
# untouched - v22kg3/v22kg4 keep using it, unchanged.
#
# Real gap found while preparing v22kg5 (the final, concise multilingual style,
# where es/it/en explanations now explicitly describe method by saying what
# was AVOIDED, e.g. "avoids unsupported scientific claims"):
# validate_kg_consistency_v3's negation guard covers "without X"/"X gabe"
# but not an "avoid(s)/avoiding unsupported X" construction - "scientific"
# is still a literal substring of "avoids unsupported scientific claims", so
# v3 would have flagged a sentence that is explicitly saying science was NOT
# used. _NEGATION_MARKERS_V4 adds that pattern; everything else (per-language
# trigger scanning, has_supported_claim_family gating, code-mixing, disease-
# mismatch, risk-level rules) is identical to v3, unchanged.
# ---------------------------------------------------------------------------
_NEGATION_MARKERS_V4 = _NEGATION_MARKERS + [
    "avoid unsupported", "avoids unsupported", "avoiding unsupported",
    "no unsupported", "not use", "does not use", "doesn't use",
]


def _is_negated_mention_v4(text_l: str, phrase_l: str) -> bool:
    segment = _segment_containing(text_l, phrase_l)
    return any(neg in segment for neg in _NEGATION_MARKERS_V4)


def validate_kg_consistency_v4(kg: dict, judge_plan: dict, counter_narrative: str = "",
                                explanation: str = "", claim_type: str = None) -> dict:
    """v22kg5/v22kg6-only. Identical to validate_kg_consistency_v3 except for one correction: the
    negation guard also recognizes "avoid(s)/avoiding unsupported X" and "no/not/doesn't use X"
    constructions - e.g. an explanation that says "avoids unsupported scientific claims" is
    explicitly NOT making a science claim, and must not be flagged as one."""
    judge_plan = judge_plan or {}
    claim_type = claim_type or (kg.get("kg_summary") or {}).get("claim_type")
    present_forbidden = {n["text"] for n in kg.get("nodes", []) if n["type"] == "forbidden_claim"}
    immutability_unsupported = not has_supported_claim_family(kg, "orientation_immutability")
    language = kg.get("language")
    langs_to_check = {language, "en"} if language else {"en"}

    def _scan(text: str):
        text_l = (text or "").lower()
        violations, triggered = [], []
        for key in present_forbidden:
            per_lang = _FORBIDDEN_KEY_TRIGGERS_BY_LANG.get(key, {})
            phrases = [p for lang in langs_to_check for p in per_lang.get(lang, [])]
            hits = [p for p in phrases if p.lower() in text_l and not _is_negated_mention_v4(text_l, p.lower())]
            if hits:
                violations.append({"forbidden_claim": key, "matched_phrases": hits})
                triggered.append(key)
        if immutability_unsupported:
            hits = [p for p in _IMMUTABILITY_CLAIM_TRIGGERS_MULTI
                    if p.lower() in text_l and not _is_negated_mention_v4(text_l, p.lower())]
            if hits:
                violations.append({"forbidden_claim": "unsupported_immutability_claim", "matched_phrases": hits})
                triggered.append("unsupported_immutability_claim")
        return violations, triggered

    plan_text = " ".join([
        str(judge_plan.get("final_response_plan") or ""),
        *[str(x) for x in (judge_plan.get("cultural_guidance") or [])],
    ])
    plan_violations, plan_triggered = _scan(plan_text)
    cn_violations, cn_triggered = _scan(counter_narrative)
    exp_violations, exp_triggered = _scan(explanation)

    if language == "ta":
        cn_code_mixing = _detect_code_mixing(counter_narrative)
        exp_code_mixing = _detect_code_mixing(explanation)
        if cn_code_mixing:
            cn_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": cn_code_mixing})
        if exp_code_mixing:
            exp_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": exp_code_mixing})

    disease_allowed = claim_type in _DISEASE_ALLOWED_CLAIM_TYPES
    if not disease_allowed:
        cn_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (counter_narrative or "").lower()]
        exp_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (explanation or "").lower()]
        if cn_disease_hits:
            cn_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": cn_disease_hits})
        if exp_disease_hits:
            exp_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": exp_disease_hits})

    cn_has_issue = bool(cn_violations)
    exp_has_issue = bool(exp_violations)
    plan_has_issue = bool(plan_violations)

    if cn_has_issue or (plan_has_issue and exp_has_issue):
        risk_level = "high"
    elif exp_has_issue:
        risk_level = "medium"
    elif plan_has_issue:
        risk_level = "low"
    else:
        risk_level = "none"

    forbidden_claims_triggered = sorted(set(plan_triggered + cn_triggered + exp_triggered))

    return {
        "final_response_plan_violations": plan_violations,
        "counter_narrative_violations": cn_violations,
        "explanation_violations": exp_violations,
        "forbidden_claims_triggered": forbidden_claims_triggered,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# v22kg7-only extensions. validate_kg_consistency_v4() above is left completely
# untouched - v22kg5/v22kg6 keep using it, unchanged.
#
# Two real gaps found in the v22kg6 English rerun: (1) v4's English science-
# trigger list only had "science"/"scientific"/"psychology" etc - it never
# had biology/brain/hormone/genetics words at all, so a cultural-only
# passage using "brain", "hormone", "biological", "chromosomes" etc could
# leak through completely unflagged; (2) the disease/claim-type-mismatch
# check only allowed disease/disorder wording when the KG's classifier
# labelled claim_type=="disease_or_pathology" - but that classifier's own
# keyword list doesn't include "insane"/"crazy" (only "disease"/"disorder"/
# "sick"/"illness"/"cure"/"mental illness"), so a comment literally saying
# "gays are insane" was NOT classified as disease_or_pathology, and a
# perfectly relevant "Homosexuality is not a mental disorder" rebuttal was
# wrongly flagged as a mismatch. Fixed here WITHOUT touching the shared
# claim_type classifier (build_counter_narrative_kg, used by every style) by
# scanning the KG's own harmful_claim/accusation node text directly for a
# wider disease/mental-illness trigger list - additive, v22kg7-only.
# ---------------------------------------------------------------------------
_V5_EXTRA_ENGLISH_SCIENCE_TRIGGERS = [
    "brain", "brain structure", "hormone", "hormones", "hormone response",
    "androgen", "androgen response", "biological", "biologically", "biology",
    "neurodevelopment", "neurodevelopmental", "genetic", "chromosomes",
]
_FORBIDDEN_KEY_TRIGGERS_BY_LANG_V5 = {
    key: {lang: list(phrases) for lang, phrases in per_lang.items()}
    for key, per_lang in _FORBIDDEN_KEY_TRIGGERS_BY_LANG.items()
}
_FORBIDDEN_KEY_TRIGGERS_BY_LANG_V5["unsupported_science_claim"]["en"] = (
    _FORBIDDEN_KEY_TRIGGERS_BY_LANG_V5["unsupported_science_claim"]["en"] + _V5_EXTRA_ENGLISH_SCIENCE_TRIGGERS
)

# Hate-comment-side trigger phrases (checked against the KG's own harmful_claim/accusation node
# text, never against a new function parameter) that mean a disease/mental-disorder rebuttal is
# actually relevant to THIS case, regardless of what the shared claim_type classifier picked.
_DISEASE_HATE_TEXT_TRIGGERS = [
    "insane", "crazy", "mentally ill", "mental illness", "mental disorder", "disorder",
    "disease", "sick", "abnormal", "defect", "cure", "fix",
    "liberated from homosexuality", "freed from homosexuality",
]


def _hate_comment_frames_disease(kg: dict) -> bool:
    harmful_texts = [n["text"] for n in kg.get("nodes", []) if n["type"] in ("harmful_claim", "accusation")]
    joined = " ".join(harmful_texts).lower()
    return any(t in joined for t in _DISEASE_HATE_TEXT_TRIGGERS)


def validate_kg_consistency_v5(kg: dict, judge_plan: dict, counter_narrative: str = "",
                                explanation: str = "", claim_type: str = None) -> dict:
    """v22kg7-only. Same evidence-conditional design and negation guard as validate_kg_
    consistency_v4 (unchanged, still used by v22kg5/v22kg6), with two corrections: (1) English
    science triggers now include biology/brain/hormone/genetics wording
    (_FORBIDDEN_KEY_TRIGGERS_BY_LANG_V5); (2) a disease/pathology rebuttal is allowed whenever
    EITHER the classifier's claim_type=="disease_or_pathology" OR the case's own harmful_claim/
    accusation KG node text frames the hate comment as disease/insane/mental-illness - fixes a
    real false positive where "gays are insane" (not classified as disease_or_pathology by the
    shared keyword classifier) wrongly flagged a relevant "not a mental disorder" rebuttal."""
    judge_plan = judge_plan or {}
    claim_type = claim_type or (kg.get("kg_summary") or {}).get("claim_type")
    present_forbidden = {n["text"] for n in kg.get("nodes", []) if n["type"] == "forbidden_claim"}
    immutability_unsupported = not has_supported_claim_family(kg, "orientation_immutability")
    language = kg.get("language")
    langs_to_check = {language, "en"} if language else {"en"}

    def _scan(text: str):
        text_l = (text or "").lower()
        violations, triggered = [], []
        for key in present_forbidden:
            per_lang = _FORBIDDEN_KEY_TRIGGERS_BY_LANG_V5.get(key, {})
            phrases = [p for lang in langs_to_check for p in per_lang.get(lang, [])]
            hits = [p for p in phrases if p.lower() in text_l and not _is_negated_mention_v4(text_l, p.lower())]
            if hits:
                violations.append({"forbidden_claim": key, "matched_phrases": hits})
                triggered.append(key)
        if immutability_unsupported:
            hits = [p for p in _IMMUTABILITY_CLAIM_TRIGGERS_MULTI
                    if p.lower() in text_l and not _is_negated_mention_v4(text_l, p.lower())]
            if hits:
                violations.append({"forbidden_claim": "unsupported_immutability_claim", "matched_phrases": hits})
                triggered.append("unsupported_immutability_claim")
        return violations, triggered

    plan_text = " ".join([
        str(judge_plan.get("final_response_plan") or ""),
        *[str(x) for x in (judge_plan.get("cultural_guidance") or [])],
    ])
    plan_violations, plan_triggered = _scan(plan_text)
    cn_violations, cn_triggered = _scan(counter_narrative)
    exp_violations, exp_triggered = _scan(explanation)

    if language == "ta":
        cn_code_mixing = _detect_code_mixing(counter_narrative)
        exp_code_mixing = _detect_code_mixing(explanation)
        if cn_code_mixing:
            cn_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": cn_code_mixing})
        if exp_code_mixing:
            exp_violations.append({"forbidden_claim": "code_mixing", "matched_phrases": exp_code_mixing})

    disease_allowed = claim_type in _DISEASE_ALLOWED_CLAIM_TYPES or _hate_comment_frames_disease(kg)
    if not disease_allowed:
        cn_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (counter_narrative or "").lower()]
        exp_disease_hits = [t for t in _DISEASE_TRIGGERS if t.lower() in (explanation or "").lower()]
        if cn_disease_hits:
            cn_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": cn_disease_hits})
        if exp_disease_hits:
            exp_violations.append({"forbidden_claim": "disease_claim_type_mismatch", "matched_phrases": exp_disease_hits})

    cn_has_issue = bool(cn_violations)
    exp_has_issue = bool(exp_violations)
    plan_has_issue = bool(plan_violations)

    if cn_has_issue or (plan_has_issue and exp_has_issue):
        risk_level = "high"
    elif exp_has_issue:
        risk_level = "medium"
    elif plan_has_issue:
        risk_level = "low"
    else:
        risk_level = "none"

    forbidden_claims_triggered = sorted(set(plan_triggered + cn_triggered + exp_triggered))

    return {
        "final_response_plan_violations": plan_violations,
        "counter_narrative_violations": cn_violations,
        "explanation_violations": exp_violations,
        "forbidden_claims_triggered": forbidden_claims_triggered,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# v22kg7 KG safety fallback - a small, deterministic, single-pass, no-LLM-call
# post-generation safety net. Not a repair loop (never calls the model
# again, never re-prompts, runs at most once per case) and not sample-
# specific (every fallback is chosen by claim-type/keyword category, never
# by dataset row ID). English only for now, per the explicit instruction to
# prefer enabling this for English initially - Spanish/Italian/Tamil/Basque
# already pass their own gates and are returned completely unchanged.
#
# Motivated by 4 real v22kg6 English findings: cultural-only brain/biology
# wording still leaked through (fixed above, in validate_kg_consistency_v5);
# mental-disorder evidence was used as the default in a child/abortion case
# instead of a direct child-dignity rebuttal; unsupported "not supported by
# science" wording appeared; and a genuinely relevant mental-disorder
# rebuttal was wrongly flagged (also fixed above). This fallback is the
# safety net for whenever the prompt-level discipline still isn't followed:
# a deterministic, pre-written, claim-type-safe answer takes over instead of
# shipping a flagged one.
# ---------------------------------------------------------------------------
_V22KG7_GENERIC_EXPLANATION = "The response directly rejects the harmful claim and emphasizes dignity and respect."

_V22KG7_ENGLISH_FALLBACKS = {
    "child_family_rejection": (
        "A child's sexual orientation is never a reason to reject them; every child deserves love and respect.",
        _V22KG7_GENERIC_EXPLANATION,
    ),
    "conversion_change_framing": (
        "Homosexuality is not a condition to be freed from; every person deserves dignity and respect.",
        _V22KG7_GENERIC_EXPLANATION,
    ),
    "same_sex_relationship_stigma": (
        "Same-sex relationships are not wrong or shameful; every relationship deserves respect when built on "
        "care and consent.",
        _V22KG7_GENERIC_EXPLANATION,
    ),
    "trans_biology_pseudoscience": (
        "Gender identity is not a defect; transgender people deserve dignity and respect.",
        _V22KG7_GENERIC_EXPLANATION,
    ),
    "disease_pathology_supported": (
        "Homosexuality is not a mental disorder; every person deserves dignity and respect.",
        _V22KG7_GENERIC_EXPLANATION,
    ),
    "disease_pathology_unsupported": (
        "Calling people disordered because of their sexual orientation is harmful; every person deserves "
        "dignity and respect.",
        _V22KG7_GENERIC_EXPLANATION,
    ),
    "generic_dignity": (
        "No one should be attacked because of who they are; everyone deserves dignity and respect.",
        _V22KG7_GENERIC_EXPLANATION,
    ),
}

# Priority-ordered, claim-type/keyword-based selection - never a sample ID, never a dataset row
# ID. Checked against the KG's own harmful_claim/accusation node text plus the raw comment text
# (belt and braces - both already describe the same hate comment).
_V22KG7_CHILD_FAMILY_KEYWORDS = [
    "abortion", "abort", "my child", "my son", "my daughter", "disown", "reject them", "reject my child",
]
_V22KG7_CONVERSION_KEYWORDS = [
    "cure", "fix them", "liberate", "liberated", "freed from", "free them from", "conversion therapy",
    "change who they are", "convert",
]
_V22KG7_TRANS_TARGET_KEYWORDS = ["trans", "transgender", "gender identity", "gender dysphoria"]
_V22KG7_TRANS_BIO_KEYWORDS = [
    "brain", "hormone", "biological", "biology", "chromosome", "genetic", "neurodevelopment",
]
_V22KG7_RELATIONSHIP_KEYWORDS = ["relationship", "dating", "couple", "marry", "marriage"]


def _select_english_kg_safety_fallback_category(kg: dict, comment: str = "") -> str:
    harmful_texts = [n["text"] for n in kg.get("nodes", []) if n["type"] in ("harmful_claim", "accusation")]
    joined = (" ".join(harmful_texts) + " " + (comment or "")).lower()

    if any(k in joined for k in _V22KG7_CHILD_FAMILY_KEYWORDS):
        return "child_family_rejection"
    if any(k in joined for k in _V22KG7_CONVERSION_KEYWORDS):
        return "conversion_change_framing"
    if any(k in joined for k in _V22KG7_TRANS_TARGET_KEYWORDS) and any(k in joined for k in _V22KG7_TRANS_BIO_KEYWORDS):
        return "trans_biology_pseudoscience"
    if any(k in joined for k in _V22KG7_RELATIONSHIP_KEYWORDS):
        return "same_sex_relationship_stigma"
    if _hate_comment_frames_disease(kg) or (comment and any(
            t in comment.lower() for t in _DISEASE_HATE_TEXT_TRIGGERS)):
        return "disease_pathology_supported" if has_supported_claim_family(kg, "science_research") else "disease_pathology_unsupported"
    return "generic_dignity"


def apply_kg_safety_fallback(kg: dict, judge_plan: dict, counter_narrative: str, explanation: str,
                              comment: str = "") -> dict:
    """v22kg7-only orchestration: validate with validate_kg_consistency_v5, then, deterministically
    and without any model call:
      - risk none/low (only final_response_plan, if anything, has violations): return output
        unchanged.
      - risk medium (explanation_violations present, counter_narrative clean): keep
        counter_narrative, replace explanation with the generic method-level fallback, revalidate.
      - risk high (counter_narrative_violations present): replace BOTH counter_narrative and
        explanation with a claim-type-safe fallback pair, revalidate.
    English only for now - other languages are returned unchanged with applied=False.
    Always returns {"counter_narrative", "explanation", "kg_safety_fallback"}, where
    kg_safety_fallback always has the same shape whether or not a fallback was applied."""
    initial_consistency = validate_kg_consistency_v5(
        kg, judge_plan, counter_narrative=counter_narrative, explanation=explanation,
    )
    risk = initial_consistency["risk_level"]

    def _no_change(reason: str) -> dict:
        return {
            "counter_narrative": counter_narrative, "explanation": explanation,
            "kg_safety_fallback": {
                "applied": False, "reason": reason,
                "original_counter_narrative": None, "original_explanation": None,
                "fallback_type": None, "post_fallback_kg_consistency": initial_consistency,
            },
        }

    if risk in ("none", "low"):
        return _no_change(
            f"risk_level={risk} - only final_response_plan (if anything) had a violation, output unchanged"
        )

    if kg.get("language") != "en":
        return _no_change(f"risk_level={risk} but fallback is scoped to English only in v22kg7")

    if risk == "medium":
        new_explanation = _V22KG7_GENERIC_EXPLANATION
        revalidated = validate_kg_consistency_v5(
            kg, judge_plan, counter_narrative=counter_narrative, explanation=new_explanation,
        )
        return {
            "counter_narrative": counter_narrative, "explanation": new_explanation,
            "kg_safety_fallback": {
                "applied": True, "reason": "explanation_violations present, counter_narrative was clean",
                "original_counter_narrative": None, "original_explanation": explanation,
                "fallback_type": "explanation_only", "post_fallback_kg_consistency": revalidated,
            },
        }

    # risk == "high"
    fallback_type = _select_english_kg_safety_fallback_category(kg, comment)
    new_cn, new_explanation = _V22KG7_ENGLISH_FALLBACKS[fallback_type]
    revalidated = validate_kg_consistency_v5(
        kg, judge_plan, counter_narrative=new_cn, explanation=new_explanation,
    )
    return {
        "counter_narrative": new_cn, "explanation": new_explanation,
        "kg_safety_fallback": {
            "applied": True, "reason": "counter_narrative_violations present (or multiple fields flagged)",
            "original_counter_narrative": counter_narrative, "original_explanation": explanation,
            "fallback_type": fallback_type, "post_fallback_kg_consistency": revalidated,
        },
    }
