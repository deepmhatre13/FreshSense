from typing import List, Optional, Dict
from pydantic import BaseModel, Field

class BoundingBoxSchema(BaseModel):
    x1: int = Field(description="Top-left X coordinate")
    y1: int = Field(description="Top-left Y coordinate")
    x2: int = Field(description="Bottom-right X coordinate")
    y2: int = Field(description="Bottom-right Y coordinate")

class ShelfLifeSchema(BaseModel):
    fruit: str = Field(description="Lower-case fruit name")
    freshness_class: str = Field(description="Freshness classification result")
    freshness_confidence: Optional[float] = Field(
        None,
        description="Freshness confidence in [0, 1]; null when the incoming confidence was unusable",
    )
    shelf_life_status: str = Field(
        description=(
            "estimated (fresh heuristic) | expired (rotten, days=0) | "
            "uncertain (freshness prediction was too uncertain) | "
            "data_not_available (no freshness model for this fruit)"
        )
    )
    remaining_days: Optional[int] = Field(None, description="Estimated remaining shelf life in days")
    typical_min_days: Optional[int] = Field(None, description="Typical minimum shelf life for this fruit")
    typical_max_days: Optional[int] = Field(None, description="Typical maximum shelf life for this fruit")
    unit: str = Field(description="Unit of time, e.g., 'days'")
    basis: str = Field(description="Reasoning basis for the estimate")
    storage_condition: str = Field(description="Assumed storage condition (ambient, refrigerated, etc.)")
    explanation: str = Field(description="Human-readable explanation of the shelf life estimate")

class FruitResultSchema(BaseModel):
    tracking_id: int = Field(description="Unique tracking ID for this object across frames")
    fruit: str = Field(description="Detected fruit name")
    freshness: str = Field(description="Stabilized freshness class (fresh, rotten, uncertain, data_not_available)")
    confidence: Optional[float] = Field(None, description="Fused detection and freshness confidence (0-1); null when freshness is data_not_available")
    detection_confidence: float = Field(description="Raw object detection confidence (0-1)")
    stabilized_confidence: float = Field(description="Stabilized freshness prediction confidence (0-1)")
    ema_confidence: float = Field(description="Exponential moving average of freshness confidence")
    is_uncertain: bool = Field(description="True if the freshness prediction is uncertain")
    is_locked: bool = Field(description="True if the prediction is locked across frames")
    lock_count: int = Field(description="Number of frames the prediction has been locked")
    majority_label: str = Field(description="Label with the most votes in the sliding window")
    vote_counts: Dict[str, int] = Field(description="Vote counts per freshness class in the current window")
    shelf_life: Optional[ShelfLifeSchema] = Field(None, description="Detailed shelf life estimation")
    bounding_box: BoundingBoxSchema = Field(description="Fruit bounding box")
    center: List[int] = Field(description="[X, Y] center coordinates of the bounding box")

class ImageMetaSchema(BaseModel):
    width: int = Field(description="Width of the processed image")
    height: int = Field(description="Height of the processed image")

class ProcessingMetaSchema(BaseModel):
    latency_ms: float = Field(description="Total inference pipeline processing time in milliseconds")
    unidentified_count: int = Field(description="Number of detected fruits that could not be classified")
    fruit_count: int = Field(description="Number of tracked fruits")

class InferenceResponse(BaseModel):
    success: bool = Field(description="Whether the inference succeeded")
    message: str = Field(description="Status message")
    image: ImageMetaSchema = Field(description="Image dimensions")
    processing: ProcessingMetaSchema = Field(description="Processing performance and stats")
    fruits: List[FruitResultSchema] = Field(description="List of detected fruits and their details")

class HealthResponse(BaseModel):
    status: str = Field(description="Overall API status (ok or error)")
    inference_ready: bool = Field(description="Whether the inference pipeline is loaded and ready")
    detector_loaded: bool = Field(description="Whether the YOLO detector is loaded")
    classifier_loaded: bool = Field(description="Whether the EfficientNet classifier is loaded")
    metadata_available: bool = Field(description="Whether fruit_database.json is available")
    device: str = Field(description="Device currently used for inference (cpu, cuda, mps)")


class DetectedObjectPreviewSchema(BaseModel):
    """A single raw YOLO detection from the lightweight preview endpoint.

    Used only for the live camera pre-capture stability loop. It deliberately
    carries NO freshness / shelf-life fields: running the full classifier every
    preview frame would be inefficient and architecturally wrong.
    """

    label: str = Field(description="Detected class name (e.g. 'Apple')")
    class_id: int = Field(description="Numeric class identifier")
    confidence: float = Field(description="Raw detection confidence in [0, 1]")
    bbox: BoundingBoxSchema = Field(description="Bounding box in pixel coordinates")


class DetectionPreviewResponse(BaseModel):
    """Response for the lightweight live-detection preview endpoint."""

    success: bool = Field(description="Whether preview detection succeeded")
    message: str = Field(description="Human-readable message")
    image: Optional[ImageMetaSchema] = Field(
        None, description="Dimensions of the preview frame (null if undecodable)"
    )
    latency_ms: float = Field(
        0.0, description="Detection-only latency in milliseconds"
    )
    detections: List[DetectedObjectPreviewSchema] = Field(
        default_factory=list, description="Detected fruit objects"
    )
