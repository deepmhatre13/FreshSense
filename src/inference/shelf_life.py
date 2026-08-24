"""Shelf-life estimation for SmartFreshAI.

WHAT THIS MODULE IS
-------------------
A deterministic, metadata-driven HEURISTIC. The estimated remaining shelf life
is derived from:

    1. the fruit's typical shelf-life range in ``fruit_database.json``
    2. the freshness state (fresh / stale / rotten / unknown / uncertain /
       unsupported)
    3. the freshness confidence (0.0-1.0)
    4. an assumed storage condition (ambient / refrigerated)

WHAT THIS MODULE IS NOT
-----------------------
It is NOT a scientifically validated time-to-event spoilage prediction model.
The output carries explicit uncertainty semantics:

* ``remaining_days`` for fresh fruit is a confidence-scaled heuristic value,
  NOT "N days with X% probability" and NOT a measured expiry date.
* ``remaining_days = None`` means "cannot be estimated" (unknown, uncertain,
  unsupported, disabled, unusable confidence).
* ``remaining_days = 0`` is reserved for expired states (rotten / stale).
* The system has NO temperature/humidity sensors and NO storage-duration
  history, so no claim about measured storage conditions is ever made. The
  storage condition is a caller-supplied ASSUMPTION that is recorded and
  echoed in the response; it does not change the numeric estimate because
  ``fruit_database.json`` contains no condition-specific durations.

See ``docs/FRESHNESS_SHELF_LIFE.md`` for the full contract.
"""

from __future__ import annotations

import logging
import math
import numbers
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from src.inference.fruit_metadata import FruitMetadata, FruitMetadataDatabase

logger = logging.getLogger(__name__)

__all__ = [
    "ShelfLifeEstimator",
    "ShelfLifeConfig",
    "ShelfLifeEstimate",
    "ALLOWED_STORAGE_CONDITIONS",
    "DEFAULT_STORAGE_CONDITION",
    "normalize_storage_condition",
    "sanitize_confidence",
]

# Storage conditions recognised by the system. These are ASSUMPTIONS supplied
# by the caller (or the configured default) -- the system has no sensors and
# never measures the real storage environment.
ALLOWED_STORAGE_CONDITIONS = ("ambient", "refrigerated")
DEFAULT_STORAGE_CONDITION = "ambient"

# Deterministic shelf-life statuses.
STATUS_DISABLED = "disabled"
STATUS_ESTIMATED = "estimated"
STATUS_EXPIRED = "expired"
STATUS_UNAVAILABLE = "unavailable"

# Stable basis identifiers (referenced by tests and documentation).
BASIS_HEURISTIC = "fruit_typical_range + freshness_state + freshness_confidence"
BASIS_METADATA_UNAVAILABLE = "metadata_unavailable"
BASIS_METADATA_INVALID = "metadata_invalid"
BASIS_FRESHNESS_UNAVAILABLE = "freshness_model_unsupported_or_uncertain"
BASIS_CONFIDENCE_UNUSABLE = "freshness_confidence_unusable"
BASIS_DISABLED = "disabled"


def normalize_storage_condition(
    value: Any,
    default: Optional[str] = None,
) -> str:
    """Validate and normalize a storage condition string.

    Args:
        value: Caller-supplied condition, or ``None`` to use ``default``.
        default: Fallback used when ``value`` is None (itself normalized).

    Returns:
        A normalized condition: ``"ambient"`` or ``"refrigerated"``.

    Raises:
        ValueError: If the value is not a supported storage condition.
            Callers (e.g. the API layer) must surface this as HTTP 400 rather
            than silently accepting arbitrary text.
    """
    if value is None:
        return normalize_storage_condition(
            DEFAULT_STORAGE_CONDITION if default is None else default
        )
    if not isinstance(value, str):
        raise ValueError(
            f"storage_condition must be one of {list(ALLOWED_STORAGE_CONDITIONS)}, "
            f"got {type(value).__name__}"
        )
    key = value.strip().lower()
    if key not in ALLOWED_STORAGE_CONDITIONS:
        raise ValueError(
            f"Unsupported storage_condition {value!r}. "
            f"Allowed values: {list(ALLOWED_STORAGE_CONDITIONS)}"
        )
    return key


def sanitize_confidence(value: Any) -> Optional[float]:
    """Coerce a confidence value into the safe range [0.0, 1.0].

    Returns:
        The clamped confidence as a float, or ``None`` when the value is
        missing or malformed (``None``, booleans, NaN, +/-infinity, strings,
        or any other non-numeric type). Malformed input must never silently
        become a meaningful-looking number: callers must treat ``None`` as
        "confidence unusable" and refuse to estimate.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, numbers.Real):
        c = float(value)
        if not math.isfinite(c):
            return None
        return max(0.0, min(1.0, c))
    # Strings and every other type are malformed, not clamped.
    return None



@dataclass(frozen=True)
class ShelfLifeConfig:
    """Configuration for shelf-life estimation.

    Attributes:
        enabled: Whether shelf-life estimation runs at all. When disabled the
            estimator reports an explicit ``disabled`` status.
        default_storage_condition: Condition assumed when a request does not
            supply one. Must be in ``ALLOWED_STORAGE_CONDITIONS``.
    """

    enabled: bool = True
    default_storage_condition: str = DEFAULT_STORAGE_CONDITION

    def __post_init__(self) -> None:
        # Fail fast on misconfiguration.
        normalize_storage_condition(self.default_storage_condition)


@dataclass(frozen=True)
class ShelfLifeEstimate:
    """Result of shelf-life estimation for ONE fruit.

    ``freshness_confidence`` is ``None`` when the incoming confidence was
    missing or malformed -- the field is never fabricated.
    """

    fruit: str
    freshness_class: str
    freshness_confidence: Optional[float]
    shelf_life_status: str
    remaining_days: Optional[int]
    typical_min_days: Optional[int]
    typical_max_days: Optional[int]
    unit: str
    basis: str
    storage_condition: str
    explanation: str

    def to_dict(self) -> dict:
        return {
            "fruit": self.fruit,
            "freshness_class": self.freshness_class,
            "freshness_confidence": (
                float(self.freshness_confidence)
                if self.freshness_confidence is not None
                else None
            ),
            "shelf_life_status": self.shelf_life_status,
            "remaining_days": self.remaining_days,
            "typical_min_days": self.typical_min_days,
            "typical_max_days": self.typical_max_days,
            "unit": self.unit,
            "basis": self.basis,
            "storage_condition": self.storage_condition,
            "explanation": self.explanation,
        }


class ShelfLifeEstimator:
    """Estimates remaining shelf life from fruit metadata + freshness confidence.

    This is a deterministic heuristic (see module docstring). It never
    invents metadata: unknown fruits, invalid ranges and unusable confidence
    all produce explicit non-estimated statuses.
    """

    def __init__(
        self,
        config: ShelfLifeConfig,
        metadata_db: Optional[FruitMetadataDatabase] = None,
    ) -> None:
        self.config = config
        self.metadata_db = metadata_db or FruitMetadataDatabase()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def estimate(
        self,
        fruit: str,
        fused_confidence: Any,
        is_fresh: Optional[bool] = None,
        freshness_class: str = "fresh",
        storage_condition: Any = None,
    ) -> ShelfLifeEstimate:
        """Estimate remaining shelf life for a single fruit.

        Args:
            fruit: Fruit name (any case; normalized internally).
            fused_confidence: Freshness confidence in ``[0, 1]``. Values
                outside the range are clamped; missing/malformed values
                (None, NaN, inf, non-numeric) yield an explicit ``uncertain``
                result instead of a fabricated estimate.
            is_fresh: Legacy boolean flag; ``False`` maps to ``rotten`` when
                no explicit freshness class is provided.
            freshness_class: One of ``fresh`` / ``stale`` / ``rotten`` /
                ``unknown`` / ``uncertain`` / ``unsupported``.
            storage_condition: Requested storage condition (``ambient`` or
                ``refrigerated``), or ``None`` for the configured default.

        Returns:
            A :class:`ShelfLifeEstimate`. ``remaining_days`` is an int only
            for ``estimated`` (fresh) and ``expired`` (rotten/stale) states;
            it is ``None`` in every non-estimated state.
        """
        key = (fruit or "").strip().lower()
        cls = (freshness_class or "").strip().lower()

        # Legacy boolean flag mapping (established project design).
        if is_fresh is False and cls == "fresh":
            cls = "rotten"

        # Storage condition is validated strictly: arbitrary text is never
        # silently accepted. None -> configured default.
        condition = normalize_storage_condition(
            storage_condition,
            default=self.config.default_storage_condition,
        )

        # Confidence is sanitized separately from the freshness state: a
        # rotten fruit stays expired even if confidence is malformed, and a
        # fresh fruit with unusable confidence must NOT get a fabricated
        # estimate.
        confidence = sanitize_confidence(fused_confidence)

        if not self.config.enabled:
            return self._result(
                key, cls, confidence, STATUS_DISABLED, None, None, None,
                BASIS_DISABLED, condition,
                "Shelf-life estimation is disabled by configuration.",
            )

        meta = self.metadata_db.get(key)

        if meta is None:
            return self._result(
                key, cls, confidence, "unsupported", None, None, None,
                BASIS_METADATA_UNAVAILABLE, condition,
                f"Fruit '{key}' has no shelf-life metadata in the database; "
                "remaining shelf life is not estimated.",
            )

        shelf_range = self._validated_range(meta)
        if shelf_range is None:
            return self._result(
                key, cls, confidence, "unsupported", None, None, None,
                BASIS_METADATA_INVALID, condition,
                f"Fruit '{key}' has invalid shelf-life metadata in the "
                "database; remaining shelf life is not estimated.",
            )
        typical_min, typical_max = shelf_range

        # ---------------- freshness state machine (deterministic) --------
        if cls == "rotten":
            return self._result(
                key, cls, confidence, STATUS_EXPIRED, 0, typical_min, typical_max,
                BASIS_HEURISTIC, condition,
                "Fruit classified as rotten; remaining shelf life is treated "
                "as expired.",
            )

        if cls == "stale":
            return self._result(
                key, cls, confidence, STATUS_EXPIRED, 0, typical_min, typical_max,
                BASIS_HEURISTIC, condition,
                "Fruit classified as stale; remaining shelf life is treated "
                "as expired.",
            )

        if cls in ("data_not_available", "unsupported", "unknown", "uncertain"):
            explanations = {
                "data_not_available": (
                    "No trained freshness model is available for this "
                    "fruit; remaining shelf life is not estimated."
                ),
                "data_not_available": (
                    "No trained freshness model is available for this "
                    "fruit; remaining shelf life is not estimated."
                ),
                "unsupported": (
                    "Freshness/shelf-life estimation is unavailable for this "
                    "fruit (no trained freshness classifier)."
                ),
                "unknown": (
                    "Freshness could not be determined reliably; remaining "
                    "shelf life is not estimated."
                ),
                "uncertain": (
                    "Freshness prediction was too unreliable; remaining "
                    "shelf life is not estimated."
                ),
            }
            return self._result(
                key, cls, confidence, cls, None, typical_min, typical_max,
                BASIS_FRESHNESS_UNAVAILABLE, condition, explanations[cls],
            )

        if cls == "fresh":
            if confidence is None:
                # Malformed confidence: never fabricate a number from it.
                return self._result(
                    key, cls, None, "uncertain", None, typical_min, typical_max,
                    BASIS_CONFIDENCE_UNUSABLE, condition,
                    "Freshness confidence was missing or malformed; "
                    "remaining shelf life cannot be estimated reliably.",
                )

            # Deterministic confidence-scaled HEURISTIC (established project
            # design): the closer the freshness confidence to 1.0, the closer
            # the estimate to the typical maximum. This is NOT a probability
            # of remaining shelf life.
            scaled_days = int(round(typical_max * confidence))
            remaining_days = max(1, min(typical_max, scaled_days))

            if confidence >= 0.8:
                tier = (
                    "Fresh fruit with high confidence; estimate is close to "
                    "maximum typical storage."
                )
            elif confidence >= 0.5:
                tier = "Fresh fruit with medium confidence; estimate scaled proportionally."
            else:
                tier = "Fresh fruit with low confidence; estimate is conservative."

            explanation = (
                f"{tier} Estimated from the fruit's typical shelf-life range "
                f"and freshness confidence. Estimate assumes {condition} storage."
            )
            return self._result(
                key, cls, confidence, STATUS_ESTIMATED, remaining_days,
                typical_min, typical_max, BASIS_HEURISTIC, condition, explanation,
            )

        # Defensive fallback for unrecognized classes: deterministic and
        # explicit, never an invented estimate.
        return self._result(
            key, cls, confidence, STATUS_UNAVAILABLE, None, typical_min, typical_max,
            BASIS_FRESHNESS_UNAVAILABLE, condition,
            f"Unrecognized freshness class '{freshness_class}'; remaining "
            "shelf life is not estimated.",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validated_range(meta: FruitMetadata) -> Optional[Tuple[int, int]]:
        """Return a validated (min, max) tuple, or None when invalid."""
        value = meta.typical_shelf_life_days
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 2
        ):
            return None
        lo, hi = value
        for bound in (lo, hi):
            if isinstance(bound, bool) or not isinstance(bound, int):
                return None
        if lo < 0 or hi <= 0 or lo > hi:
            return None
        return int(lo), int(hi)

    @staticmethod
    def _result(
        fruit: str,
        freshness_class: str,
        confidence: Optional[float],
        status: str,
        remaining_days: Optional[int],
        typical_min: Optional[int],
        typical_max: Optional[int],
        basis: str,
        storage_condition: str,
        explanation: str,
    ) -> ShelfLifeEstimate:
        return ShelfLifeEstimate(
            fruit=fruit,
            freshness_class=freshness_class,
            freshness_confidence=confidence,
            shelf_life_status=status,
            remaining_days=remaining_days,
            typical_min_days=typical_min,
            typical_max_days=typical_max,
            unit="days",
            basis=basis,
            storage_condition=storage_condition,
            explanation=explanation,
        )


