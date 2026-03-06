"""
similarity_tiers.py
-------------------
Centralised logic for interpreting cosine similarity scores into
human-readable tiers. Used by the upload pipeline and the files API.
"""

# ── Tier thresholds ───────────────────────────────────────────────────────────

EXACT_THRESHOLD = 1.00        # handled by hash; cosine will be ~1.0
NEAR_DUPLICATE_THRESHOLD = 0.90
RELATED_THRESHOLD = 0.70


# ── Tier definitions ──────────────────────────────────────────────────────────

TIERS = {
    "exact":        {"emoji": "🔴", "label": "Exact Duplicate",  "action": "Safe to delete"},
    "near":         {"emoji": "🟡", "label": "Near Duplicate",   "action": "Review before deleting"},
    "related":      {"emoji": "🟢", "label": "Related File",     "action": "Grouped — not a duplicate"},
    "unique":       {"emoji": "⚪", "label": "Unique",           "action": None},
}


def get_tier(similarity_score: float | None, is_exact_duplicate: bool = False) -> dict:
    """
    Given a cosine similarity score (0.0–1.0) and whether an exact hash
    match was found, return the appropriate tier dict:

    {
        "tier":    "near",
        "emoji":   "🟡",
        "label":   "Near Duplicate",
        "action":  "Review before deleting",
        "percent": "97.4%"
    }
    """
    if is_exact_duplicate:
        tier = TIERS["exact"].copy()
        tier["tier"] = "exact"
        tier["percent"] = "100.0%"
        return tier

    if similarity_score is None:
        tier = TIERS["unique"].copy()
        tier["tier"] = "unique"
        tier["percent"] = None
        return tier

    pct = f"{similarity_score * 100:.1f}%"

    if similarity_score >= NEAR_DUPLICATE_THRESHOLD:
        tier = TIERS["near"].copy()
        tier["tier"] = "near"
        tier["percent"] = pct
        return tier

    if similarity_score >= RELATED_THRESHOLD:
        tier = TIERS["related"].copy()
        tier["tier"] = "related"
        tier["percent"] = pct
        return tier

    tier = TIERS["unique"].copy()
    tier["tier"] = "unique"
    tier["percent"] = pct
    return tier


def tier_summary(tier_dict: dict) -> str:
    """
    Return a single human-readable summary string, e.g.:
    '🟡 Near Duplicate (97.4%) — Review before deleting'
    """
    parts = [f"{tier_dict['emoji']} {tier_dict['label']}"]
    if tier_dict.get("percent"):
        parts[0] += f" ({tier_dict['percent']})"
    if tier_dict.get("action"):
        parts.append(tier_dict["action"])
    return " — ".join(parts)
