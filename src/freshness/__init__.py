"""Freshness availability registry — SINGLE SOURCE OF TRUTH.

This module is the ONLY place that decides whether a fruit has a validated
freshness ML model. It reads ``configs/freshness_availability.json`` and never
duplicates that decision in pipeline files, the API, the webcam normalizer, or
tests.

The production freshness contract uses four controlled values:

    "fresh"  | "rotten" | "uncertain" | "data_not_available"

* ``data_not_available`` is returned whenever the fruit has no validated
  freshness model (``NOT_AVAILABLE``) or exists only as detector-unsupported
  training data (``AVAILABLE_DETECTOR_UNSUPPORTED``), i.e. a fruit the frozen
  YOLO detector cannot emit.
* A fruit marked ``AVAILABLE`` has a real 16-class model and a real prediction
  may be ``fresh`` / ``rotten`` / ``uncertain``.

Never guess freshness from image rules, from a fruit's colour, from YOLO
confidence, or from a fruit name. Report ``data_not_available`` instead.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import FrozenSet, Set

logger = logging.getLogger(__name__)

# Controlled output vocabulary.
FRESH = "fresh"
ROTTEN = "rotten"
UNCERTAIN = "uncertain"
DATA_NOT_AVAILABLE = "data_not_available"
FRESHNESS_VOCABULARY = {FRESH, ROTTEN, UNCERTAIN, DATA_NOT_AVAILABLE}

# Registry statuses.
AVAILABLE = "AVAILABLE"
AVAILABLE_DETECTOR_UNSUPPORTED = "AVAILABLE_DETECTOR_UNSUPPORTED"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
NOT_AVAILABLE = "NOT_AVAILABLE"
# Canonical 16-class freshness taxonomy derived ONLY from fruits with valid
# fresh + rotten training data in ``data/Original Image``. Class ids are
# deterministic and match ``data/freshness/class_mapping.json``.
_CANONICAL_CLASSES: list[str] = [
    "Apple_fresh",
    "Apple_rotten",
    "banana_fresh",
    "banana_rotten",
    "Grape_fresh",
    "Grape_rotten",
    "guava_fresh",
    "guava_rotten",
    "Jujube_fresh",
    "Jujube_rotten",
    "Orange_fresh",
    "Orange_rotten",
    "Pomegranate_fresh",
    "Pomegranate_rotten",
    "Strawberry_fresh",
    "Strawberry_rotten",
]

CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(_CANONICAL_CLASSES)}
ID_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(_CANONICAL_CLASSES)}


def _default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "freshness_availability.json"


class FreshnessAvailability:
    """Registry-backed availability lookup.

    Loads ``configs/freshness_availability.json`` once and exposes a small,
    typed API. The JSON is the single source of truth; this wrapper only makes
    it convenient to consume.
    """

    def __init__(self, registry_path: str | Path | None = None) -> None:
        self.path = Path(registry_path) if registry_path else _default_registry_path()
        self._entries: dict[str, dict] = {}
        self._by_key: dict[str, dict] = {}
        self.load()

    def load(self) -> None:
        """Load and validate the registry. Never raises on a bad file: a
        missing registry degrades safely to ``NOT_AVAILABLE`` for every fruit."""
        if not self.path.exists():
            logger.error("Freshness availability registry missing: %s", self.path)
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            logger.error("Freshness availability registry invalid JSON: %s", exc)
            return
        fruits = raw.get("fruits", {}) if isinstance(raw, dict) else {}
        if not isinstance(fruits, dict):
            return
        self._entries = fruits
        self._by_key = {k.strip().lower(): v for k, v in fruits.items()}
# ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self, fruit: str | None) -> str:
        """Return the registry status for ``fruit`` (case-insensitive)."""
        key = (fruit or "").strip().lower()
        entry = self._by_key.get(key)
        if not entry:
            return NOT_AVAILABLE
        return str(entry.get("availability", NOT_AVAILABLE))

    def is_available(self, fruit: str | None) -> bool:
        """True only for fruits with a *production-usable* freshness model.

        ``AVAILABLE`` means a freshness prediction can be emitted.
        ``AVAILABLE_DETECTOR_UNSUPPORTED`` still has training data but the
        frozen detector cannot surface the fruit, so production inference
        reports ``data_not_available`` (the data is real, the detector cannot
        deliver it).
        """
        return self.status(fruit) == AVAILABLE

    def freshness_value(self, fruit: str | None) -> str:
        """The freshness vocabulary value production must return for a fruit.

        Returns ``DATA_NOT_AVAILABLE`` for any fruit without a usable model.
        """
        if self.is_available(fruit):
            return UNCERTAIN  # place-holder; real callers resolve fresh/rotten from the model
        return DATA_NOT_AVAILABLE

    def class_id(self, fruit: str | None) -> set[str] | None:
        key = (fruit or "").strip().lower()
        entry = self._by_key.get(key)
        if not entry or entry.get("class_id") is None:
            return None
        cid = str(entry["class_id"])
        return {c for c in cid.split("/") if c}

    def supported_fruits(self) -> Set[str]:
        """All fruits whose availability status is ``AVAILABLE``."""
        return {
            k
            for k, v in self._by_key.items()
            if str(v.get("availability")) == AVAILABLE
        }

    def canonical_classes(self) -> list[str]:
        """The 16-class deterministic taxonomy."""
        return list(_CANONICAL_CLASSES)

    def num_classes(self) -> int:
        return len(_CANONICAL_CLASSES)


# A module-level default instance so consumers can simply import it.
_default_registry: FreshnessAvailability | None = None


def get_registry() -> FreshnessAvailability:
    """Return the process-wide default registry (lazily initialised)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = FreshnessAvailability()
    return _default_registry


def freshness_supported(fruit: str | None) -> bool:
    """Back-compat helper: whether ``fruit`` has a usable freshness model."""
    return get_registry().is_available(fruit)


def freshness_value(fruit: str | None) -> str:
    """Return the controlled freshness vocabulary value for a fruit."""
    return get_registry().freshness_value(fruit)


__all__ = [
    "AVAILABLE",
    "AVAILABLE_DETECTOR_UNSUPPORTED",
    "DATA_NOT_AVAILABLE",
    "FRESH",
    "FRESHNESS_VOCABULARY",
    "FreshnessAvailability",
    "INSUFFICIENT_DATA",
    "NOT_AVAILABLE",
    "ROTTEN",
    "UNCERTAIN",
    "CLASS_TO_ID",
    "ID_TO_CLASS",
    "freshness_supported",
    "freshness_value",
    "get_registry",
]