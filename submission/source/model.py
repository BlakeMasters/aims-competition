"""Adaptive-label-anchored runtime for the Predictive Evaluation Challenge.

Key idea (validated offline): the dominant transferable signal on hidden,
unseen benchmarks is the *per-category base rate*, which the platform reveals
through the K labeled examples passed to ``predict``. Rather than nudging a
fixed center toward those labels (the previous -0.62 family), this runtime
estimates a hierarchical, shrunk base rate from the revealed labels and predicts
close to it, with a light subject-ability adjustment. When no labels are
available it falls back to the proven compact domain-prior formula.
"""

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
SUBJECT_FAMILY_STATS: dict[str, dict[str, float]] = {}
FAMILY_MEAN_STATS: dict[str, dict[str, float]] = {}
DOMAIN_FAMILY_STATS: dict[str, dict[str, float]] = {}
DOMAIN_SIGNATURE_STATS: dict[str, dict[str, float]] = {}
DOMAIN_SHAPE_STATS: dict[str, dict[str, float]] = {}
FORMULA = {"center": 0.6, "alias_weight": 0.15, "domain_weight": 0.30, "strength": 1.0}

# Anchoring configuration (overridable from the artifact's "anchor" block).
ANCHOR = {
    "enabled": True,
    "alpha": 1.0,          # Laplace pseudo-count for level mean estimation
    "shrink": 4.0,         # logit-space shrink: w = n / (n + shrink)
    "beta_subject": 0.15,  # weight on centered subject-ability logit
    "label_trust": 1.0,    # blend between anchor (1.0) and fixed-formula prior
    "min_pool": 1,         # minimum revealed labels before trusting the anchor
    "levels": ["family", "signature", "benchmark", "benchcond"],
    "subject_cap": 2.5,    # cap on |centered subject logit| to avoid outliers
    "family_subject": True,    # use per-(subject, family) ability when available
    "family_subject_min": 30,  # min observations for a family-specific offset
}


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(correct) as a finite native Python float."""
    try:
        subject = str(input.get("subject_content", "") or "")
        item = str(input.get("item_content", "") or "")
        benchmark = _normalize_text(input.get("benchmark", ""))
        condition = _normalize_text(input.get("condition", "none"))
        alias = _normalize_alias(_extract_subject_name(subject))

        prior = _formula_prior(alias, item)

        if ANCHOR.get("enabled", True) and labeled:
            anchored = _anchored_probability(
                alias, benchmark, condition, item, prior, labeled
            )
            if anchored is not None:
                return _clip_probability(anchored)

        # Fallback: previous compact formula nudged by labels (robust floor).
        p = _adjust_with_labeled(prior, alias, benchmark, condition, labeled)
        return _clip_probability(p)
    except Exception:
        return _clip_probability(GLOBAL_MEAN)


# --------------------------------------------------------------------------
# Anchoring core (importable + unit-validated separately)
# --------------------------------------------------------------------------

def _anchored_probability(alias, benchmark, condition, item, prior, labeled):
    keys = _item_keys(item, benchmark, condition)
    records = _parse_labeled(labeled)
    if not records:
        return None
    pooled = [v for _, v, _ in records]
    if len(pooled) < int(ANCHOR.get("min_pool", 1)):
        return None

    alpha = float(ANCHOR.get("alpha", 1.0))
    shrink = float(ANCHOR.get("shrink", 4.0))
    beta = float(ANCHOR.get("beta_subject", 0.0))
    cap = float(ANCHOR.get("subject_cap", 2.5))
    debias = bool(ANCHOR.get("debias_subject", False)) and beta > 0.0

    def level_estimate(obs_vals, obs_offsets):
        # category difficulty in logit space; if de-biasing, remove the mean
        # subject offset of the revealed sample so a strong/weak subject mix in
        # the K labels does not skew the category base rate.
        ll = _smoothed_logit(sum(obs_vals), len(obs_vals), alpha)
        if debias and obs_offsets:
            ll -= beta * (sum(obs_offsets) / len(obs_offsets))
        return ll

    pooled_offsets = [o for _, _, o in records]
    pooled_vals = [v for _, v, _ in records]
    level_logit = level_estimate(pooled_vals, pooled_offsets)
    matched = False
    # coarse -> fine: each finer level that has support shrinks toward the
    # running (coarser) estimate.
    for level in ANCHOR.get("levels", []):
        target_key = keys.get(level)
        if not target_key:
            continue
        obs_vals = [v for k, v, o in records if k.get(level) == target_key]
        if not obs_vals:
            continue
        obs_offsets = [o for k, v, o in records if k.get(level) == target_key]
        matched = True
        ml = level_estimate(obs_vals, obs_offsets)
        w = len(obs_vals) / (len(obs_vals) + shrink)
        level_logit = w * ml + (1.0 - w) * level_logit

    anchor_logit = level_logit
    if beta:
        target_offset = _subject_offset(alias, cap, keys.get("family"))
        if target_offset is not None:
            anchor_logit += beta * target_offset

    p_anchor = _sigmoid(anchor_logit)
    trust = float(ANCHOR.get("label_trust", 1.0))
    if not matched:
        # Only the pooled mean was usable; trust it a bit less.
        trust *= 0.85
    trust = min(max(trust, 0.0), 1.0)
    return trust * p_anchor + (1.0 - trust) * prior


_PARSE_CACHE: dict = {"key": None, "records": None}


def _parse_labeled(labeled):
    # The hosted runtime passes the same `labeled` list to every predict() call
    # in a round; parse it once (5000 calls x hundreds of labels otherwise).
    cache_key = (id(labeled), len(labeled) if labeled else 0)
    if _PARSE_CACHE["key"] == cache_key:
        return _PARSE_CACHE["records"]
    records = []
    for row in labeled or []:
        try:
            visible = row.get("input", row)
            raw = row.get("label", row.get("response", row.get("correct")))
            label = float(raw)
            if not math.isfinite(label):
                continue
            value = 1.0 if label >= 0.5 else 0.0
            keys = _item_keys(
                str(visible.get("item_content", "") or ""),
                _normalize_text(visible.get("benchmark", "")),
                _normalize_text(visible.get("condition", "none")),
            )
            alias = _normalize_alias(
                _extract_subject_name(str(visible.get("subject_content", "") or ""))
            )
            keys["subject"] = alias
            offset = _subject_offset(
                alias, float(ANCHOR.get("subject_cap", 2.5)), keys.get("family")
            ) or 0.0
            records.append((keys, value, offset))
        except Exception:
            continue
    _PARSE_CACHE["key"] = cache_key
    _PARSE_CACHE["records"] = records
    return records


def _subject_offset(alias, cap, family=None):
    # Family-relative subject ability when available; else global-relative.
    if family and ANCHOR.get("family_subject", True):
        info = SUBJECT_FAMILY_STATS.get(f"{alias}||{family}")
        if info and int(info.get("count", 0)) >= int(ANCHOR.get("family_subject_min", 30)):
            try:
                sf = float(info["mean"])
                fm = _lookup_mean(FAMILY_MEAN_STATS, family, GLOBAL_MEAN)
                centered = _safe_logit(sf) - _safe_logit(fm)
                return max(-cap, min(cap, centered))
            except Exception:
                pass
    subj = _lookup_mean(SUBJECT_ALIAS_STATS, alias, None)
    if subj is None:
        return None
    centered = _safe_logit(subj) - _safe_logit(GLOBAL_MEAN)
    return max(-cap, min(cap, centered))


def _item_keys(item, benchmark, condition):
    dk = _domain_keys(item)
    return {
        "family": dk["family"],
        "signature": dk["signature"],
        "benchmark": benchmark or "",
        "benchcond": f"{benchmark}||{condition}" if benchmark else "",
    }


def _smoothed_logit(positive, total, alpha):
    p = (positive + alpha) / (total + 2.0 * alpha)
    return _safe_logit(p)


# --------------------------------------------------------------------------
# Fallback formula prior (compact domain-prior, no labels)
# --------------------------------------------------------------------------

def _formula_prior(alias, item):
    center = FORMULA.get("center")
    center_value = GLOBAL_MEAN if center is None else _clip_probability(center)
    alias_mean = _lookup_mean(SUBJECT_ALIAS_STATS, alias, GLOBAL_MEAN)
    domain = _domain_prior(item)
    p = center_value + float(FORMULA.get("strength", 1.0)) * (
        float(FORMULA.get("alias_weight", 0.0)) * (alias_mean - center_value)
        + float(FORMULA.get("domain_weight", 0.0)) * (domain - center_value)
    )
    return _clip_probability(p)


def _adjust_with_labeled(p, alias, benchmark, condition, labeled):
    if not labeled:
        return p
    records = _parse_labeled(labeled)
    if not records:
        return p
    all_values = [v for _, v, _ in records]
    same_subject = [v for k, v, _ in records if k.get("subject") == alias]
    same_bench = [v for k, v, _ in records if k.get("benchmark") == benchmark]
    same_cond = [v for k, v, _ in records if k.get("benchcond", "").endswith(f"||{condition}")]
    p = _label_shift(p, all_values, GLOBAL_MEAN, 0.30, 10.0)
    p = _label_shift(p, same_subject, GLOBAL_MEAN, 0.28, 4.0)
    p = _label_shift(p, same_bench, GLOBAL_MEAN, 0.18, 6.0)
    p = _label_shift(p, same_cond, GLOBAL_MEAN, 0.10, 8.0)
    return p


def _label_shift(p, values, center, max_weight, shrink):
    if not values:
        return p
    observed = sum(values) / len(values)
    weight = max_weight * (len(values) / (len(values) + shrink))
    return p + weight * (observed - center)


# --------------------------------------------------------------------------
# Domain signatures
# --------------------------------------------------------------------------

def _domain_prior(item):
    keys = _domain_keys(item)
    shape = _lookup_mean(DOMAIN_SHAPE_STATS, keys["shape"], GLOBAL_MEAN)
    family = _lookup_mean(DOMAIN_FAMILY_STATS, keys["family"], shape)
    signature = _lookup_mean(DOMAIN_SIGNATURE_STATS, keys["signature"], family)
    return _clip_probability(0.50 * signature + 0.30 * family + 0.20 * shape)


def _domain_keys(item):
    text = str(item or "")
    lower = text.lower()
    tokens = TOKEN_RE.findall(text)
    option_count = len(OPTION_RE.findall(text))
    number_count = len(NUMBER_RE.findall(text))
    flags = []
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


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _load_artifact():
    global EPS, GLOBAL_MEAN, SUBJECT_ALIAS_STATS, SUBJECT_FAMILY_STATS, FAMILY_MEAN_STATS
    global DOMAIN_FAMILY_STATS, DOMAIN_SIGNATURE_STATS, DOMAIN_SHAPE_STATS, FORMULA, ANCHOR
    if not ARTIFACT_PATH.exists():
        return
    with ARTIFACT_PATH.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    EPS = float(payload.get("eps", EPS))
    GLOBAL_MEAN = _clip_probability(payload.get("global_mean", GLOBAL_MEAN))
    SUBJECT_ALIAS_STATS = payload.get("subject_alias_stats", {}) or {}
    SUBJECT_FAMILY_STATS = payload.get("subject_family_stats", {}) or {}
    FAMILY_MEAN_STATS = payload.get("family_mean_stats", {}) or {}
    DOMAIN_FAMILY_STATS = payload.get("domain_family_stats", {}) or {}
    DOMAIN_SIGNATURE_STATS = payload.get("domain_signature_stats", {}) or {}
    DOMAIN_SHAPE_STATS = payload.get("domain_shape_stats", {}) or {}
    FORMULA.update(payload.get("runtime_formula", {}) or {})
    ANCHOR.update(payload.get("anchor", {}) or {})


def _extract_subject_name(subject_content):
    for line in str(subject_content or "").splitlines():
        if line.lower().startswith("name:"):
            return line.split(":", 1)[1]
    return ""


def _lookup_mean(stats, key, fallback):
    info = stats.get(str(key))
    if not info:
        return fallback
    try:
        return float(info.get("mean", fallback))
    except Exception:
        return fallback


def _normalize_text(text):
    return " ".join(str(text or "").lower().strip().split())


def _normalize_alias(text):
    value = str(text or "").lower()
    value = value.replace("instruct", "inst")
    value = value.replace("chat", "")
    return ALIAS_RE.sub("", value)


def _safe_logit(value):
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


def _sigmoid(value):
    value = min(max(float(value), -30.0), 30.0)
    return 1.0 / (1.0 + math.exp(-value))


def _clip_probability(value):
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
