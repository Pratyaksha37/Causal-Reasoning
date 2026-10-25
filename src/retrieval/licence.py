"""Licence checking and terms-of-use compliance for Europe PMC records."""

from typing import Any


def assess_licence(record: dict[str, Any]) -> dict[str, Any]:
    """
    Assess licensing status from a raw Europe PMC record.

    Returns:
        Dict with 'licence_note' (str) and 'can_store_full_text' (bool).
    """
    is_oa = record.get("isOpenAccess", "N") == "Y"
    license_str = (record.get("license") or "").strip().lower()

    # Check for Creative Commons permissive licenses that allow storage/redistribution
    permissive_tokens = ["cc by", "cc-by", "cc0", "public domain", "open access"]
    is_permissive = any(token in license_str for token in permissive_tokens)

    can_store = is_oa and is_permissive

    note = license_str if license_str else ("Open Access" if is_oa else "abstract-only")

    return {
        "licence_note": note,
        "can_store_full_text": can_store,
    }
