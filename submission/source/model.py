"""Compact domain-prior runtime for the Predictive Evaluation Challenge."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path


ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "domain_prior.json"
TOKEN_RE = re.compile(r"[A-Za-z0-9_./:+-]{2,}")
ALIAS_RE = re.compile(r"[^a-z0-9]+")
OPTION_RE = re.compile(r"(?:^|\n|\s)(?:\(?[A-Ha-h]\)?[\).:]|option\s+[A-Ha-h]\b)")
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
CODE_RE = re.compile(r"\b(def|class|import|return|function|public|private|const|let|var|python|java|javascript|sql|bash|stack trace|exception)\b|```|[{};]")
MATH_RE = re.compile(r"\b(prove|theorem|equation|integral|derivative|matrix|geometry|probability|calculate|solve|integer|decimal)\b|[=+\-*/^<>]")
MEDICAL_RE = re.compile(r"\b(patient|diagnosis|clinical|treatment|symptom|disease|drug|dose|medical|doctor|hospital|therapy)\b")
SAFETY_RE = re.compile(r"\b(safety|harm|attack|exploit|malware|toxic|bias|jailbreak|policy|privacy|secure|risk)\b")
TOOL_RE = re.compile(r"\b(api|tool|function call|json|browser|android|web|terminal|shell|agent|environment)\b")
VISUAL_RE = re.compile(r"\b(image|diagram|figure|chart|graph|picture|visual|screenshot|bounding box)\b")

EPS = 1e-4
GLOBAL_MEAN = 0.5
SUBJECT_ALIAS_STATS: dict[str, dict[str, float]] = {}
DOMAIN_FAMILY_STATS: dict[str, dict[str, float]] = {}
DOMAIN_SIGNATURE_STATS: dict[str, dict[str, float]] = {}
DOMAIN_SHAPE_STATS: dict[str, dict[str, float]] = {}
FORMULA = {"center": None, "alias_weight": 0.10, "domain_weight": 0.30, "strength": 1.0}


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) as a finite native Python float."""
    try:
        subject = str(input.get("subject_content", "") or "")
        item = str(input.get("item_content", "") or "")
        center = FORMULA.get("center")
        center_value = GLOBAL_MEAN if center is None else _clip_probability(center)
        alias = _lookup_mean(SUBJECT_ALIAS_STATS, _normalize_alias(_extract_subject_name(subject)), GLOBAL_MEAN)
        domain = _domain_prior(item)
        p = center_value + float(FORMULA.get("strength", 1.0)) * (
            float(FORMULA.get("alias_weight", 0.0)) * (alias - center_value)
            + float(FORMULA.get("domain_weight", 0.0)) * (domain - center_value)
        )
        p = _adjust_with_labeled(_clip_probability(p), subject, input, labeled)
        return _clip_probability(p)
    except Exception:
        return _clip_probability(GLOBAL_MEAN)


def _load_artifact() -> None:
    global EPS, GLOBAL_MEAN, SUBJECT_ALIAS_STATS, DOMAIN_FAMILY_STATS
    global DOMAIN_SIGNATURE_STATS, DOMAIN_SHAPE_STATS, FORMULA
    if not ARTIFACT_PATH.exists():
        return
    with ARTIFACT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    EPS = float(payload.get("eps", EPS))
    GLOBAL_MEAN = _clip_probability(payload.get("global_mean", GLOBAL_MEAN))
    SUBJECT_ALIAS_STATS = payload.get("subject_alias_stats", {}) or {}
    DOMAIN_FAMILY_STATS = payload.get("domain_family_stats", {}) or {}
    DOMAIN_SIGNATURE_STATS = payload.get("domain_signature_stats", {}) or {}
    DOMAIN_SHAPE_STATS = payload.get("domain_shape_stats", {}) or {}
    FORMULA.update(payload.get("runtime_formula", {}) or {})


def _domain_prior(item: str) -> float:
    keys = _domain_keys(item)
    shape = _lookup_mean(DOMAIN_SHAPE_STATS, keys["shape"], GLOBAL_MEAN)
    family = _lookup_mean(DOMAIN_FAMILY_STATS, keys["family"], shape)
    signature = _lookup_mean(DOMAIN_SIGNATURE_STATS, keys["signature"], family)
    return _clip_probability(0.50 * signature + 0.30 * family + 0.20 * shape)


def _domain_keys(item: str) -> dict[str, str]:
    text = str(item or "")
    lower = text.lower()
    tokens = TOKEN_RE.findall(text)
    option_count = len(OPTION_RE.findall(text))
    number_count = len(NUMBER_RE.findall(text))
    flags: list[str] = []
    if CODE_RE.search(lower):
        flags.append("code")
    if MATH_RE.search(lower):
        flags.append("math")
    if MEDICAL_RE.search(lower):
        flags.append("medical")
    if SAFETY_RE.search(lower):
        flags.append("safety")
    if TOOL_RE.search(lower):
        flags.append("tool")
    if VISUAL_RE.search(lower):
        flags.append("visual")
    if option_count >= 4:
        flags.append("multi_choice")
    if not flags:
        flags.append("generic")

    if "code" in flags:
        family = "code"
    elif "tool" in flags:
        family = "tool"
    elif "medical" in flags:
        family = "medical"
    elif "safety" in flags:
        family = "safety"
    elif "visual" in flags and "math" in flags:
        family = "visual_math"
    elif "visual" in flags:
        family = "visual"
    elif "math" in flags:
        family = "math"
    elif "multi_choice" in flags:
        family = "multi_choice"
    else:
        family = "generic"

    if len(tokens) < 40:
        length = "short"
    elif len(tokens) < 120:
        length = "medium"
    elif len(tokens) < 320:
        length = "long"
    else:
        length = "very_long"
    options = "opt4" if option_count >= 4 else ("opt1to3" if option_count else "opt0")
    numbers = "num_many" if number_count >= 6 else ("num_some" if number_count else "num0")
    flag_key = "+".join(sorted(flags))
    return {
        "family": family,
        "shape": f"{length}|{options}|{numbers}",
        "signature": f"{family}|{length}|{options}|{numbers}|{flag_key}",
    }


def _adjust_with_labeled(p: float, subject_content: str, input_row: dict, labeled: list[dict] | None) -> float:
    if not labeled:
        return p
    all_values: list[float] = []
    same_subject: list[float] = []
    subject_alias = _normalize_alias(_extract_subject_name(subject_content))
    benchmark = _normalize_text(input_row.get("benchmark", ""))
    condition = _normalize_text(input_row.get("condition", "none"))
    same_benchmark: list[float] = []
    same_condition: list[float] = []
    for row in labeled:
        try:
            visible = row.get("input", row)
            label = float(row.get("label", row.get("response", row.get("correct"))))
            if not math.isfinite(label):
                continue
            value = 1.0 if label >= 0.5 else 0.0
            all_values.append(value)
            if _normalize_alias(_extract_subject_name(str(visible.get("subject_content", "") or ""))) == subject_alias:
                same_subject.append(value)
            if _normalize_text(visible.get("benchmark", "")) == benchmark:
                same_benchmark.append(value)
            if _normalize_text(visible.get("condition", "none")) == condition:
                same_condition.append(value)
        except Exception:
            continue
    adjusted = p
    adjusted = _label_shift(adjusted, all_values, GLOBAL_MEAN, 0.30, 10.0)
    adjusted = _label_shift(adjusted, same_subject, GLOBAL_MEAN, 0.28, 4.0)
    adjusted = _label_shift(adjusted, same_benchmark, GLOBAL_MEAN, 0.18, 6.0)
    adjusted = _label_shift(adjusted, same_condition, GLOBAL_MEAN, 0.10, 8.0)
    return adjusted


def _label_shift(p: float, values: list[float], center: float, max_weight: float, shrink: float) -> float:
    if not values:
        return p
    observed = sum(values) / len(values)
    weight = max_weight * (len(values) / (len(values) + shrink))
    return p + weight * (observed - center)


def _extract_subject_name(subject_content: str) -> str:
    for line in str(subject_content or "").splitlines():
        if line.lower().startswith("name:"):
            return line.split(":", 1)[1]
    return ""


def _lookup_mean(stats: dict[str, dict[str, float]], key: object, fallback: float) -> float:
    info = stats.get(str(key))
    if not info:
        return float(fallback)
    try:
        return float(info.get("mean", fallback))
    except Exception:
        return float(fallback)


def _normalize_text(text: object) -> str:
    return " ".join(str(text or "").lower().strip().split())


def _normalize_alias(text: object) -> str:
    value = str(text or "").lower()
    value = value.replace("instruct", "inst")
    value = value.replace("chat", "")
    return ALIAS_RE.sub("", value)


def _clip_probability(value: object) -> float:
    try:
        p = float(value)
    except Exception:
        return float(GLOBAL_MEAN)
    if not math.isfinite(p):
        return float(GLOBAL_MEAN)
    if p < EPS:
        return float(EPS)
    if p > 1.0 - EPS:
        return float(1.0 - EPS)
    return float(p)


_load_artifact()
