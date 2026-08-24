import logging
import time
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

from src.api.schemas import (
    InferenceResponse,
    HealthResponse,
    ImageMetaSchema,
    ProcessingMetaSchema,
    FruitResultSchema,
    DetectionPreviewResponse,
    DetectedObjectPreviewSchema,
    BoundingBoxSchema,
)
from src.inference.detection_pipeline import DetectionPipeline, DetectionPipelineConfig
from src.inference.predictor import Predictor
from src.inference.shelf_life import ShelfLifeConfig, normalize_storage_condition
from src.inference.fruit_metadata import FruitMetadataDatabase
from configs.config import Config

logger = logging.getLogger(__name__)

# Global state to hold pipeline
pipeline: Optional[DetectionPipeline] = None
predictor: Optional[Predictor] = None
metadata_available: bool = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, predictor, metadata_available
    
    logger.info("Initializing SmartFreshAI API...")
    
    # 0. Load shelf-life configuration (heuristic policy only; the fruit
    #    shelf-life numbers themselves live in fruit_database.json).
    shelf_life_config = ShelfLifeConfig()
    try:
        yaml_config = Config.from_yaml("configs/settings.yaml")
        sl = yaml_config.shelf_life
        shelf_life_config = ShelfLifeConfig(
            enabled=sl.enabled,
            default_storage_condition=sl.default_storage_condition,
        )
        logger.info(
            "Shelf-life estimator ready (enabled=%s, default_storage_condition=%s).",
            shelf_life_config.enabled,
            shelf_life_config.default_storage_condition,
        )
    except Exception as exc:
        logger.warning(
            "Could not load configs/settings.yaml (%s); using default ShelfLifeConfig.", exc
        )
    
    # 0b. Fruit metadata availability (single source of truth:
    #     fruit_database.json). Missing/invalid metadata never crashes the
    #     app; the estimator reports an explicit "unsupported" status.
    metadata_db = FruitMetadataDatabase()
    metadata_available = metadata_db.metadata_available
    if not metadata_available:
        logger.warning("Fruit metadata unavailable; shelf-life will report 'unsupported'.")
    elif metadata_db.validation_issues:
        logger.warning("Fruit metadata issues: %s", metadata_db.validation_issues)
    
    # 1. Load Predictor (EfficientNet-B0 freshness classifier)
    #    Load the versioned 16-class checkpoint that covers every fruit with
    #    legitimate fresh/rotten training data. The old best_model.pth is
    #    treated as an immutable baseline -- it is NEVER overwritten here
    #    (promotion happens via explicit evaluate-and-swap).
    candidate_paths = [
        "models/checkpoints/freshness_efficientnet_b0_16class.pth",
        "models/checkpoints/best_model.pth",
    ]
    predictor = None
    for checkpoint_path in candidate_paths:
        if Path(checkpoint_path).exists():
            try:
                predictor = Predictor(checkpoint_path=checkpoint_path)
                logger.info("Predictor loaded from %s", checkpoint_path)
                break
            except Exception as e:
                logger.error(f"Failed to load Predictor from {checkpoint_path}: {e}")
    if predictor is None:
        logger.warning(
            "No freshness checkpoint found. Freshness will be reported as "
            "'data_not_available' for all fruits."
        )

    # 2. Metadata DB availability is already established above (step 0b).
    #    ``metadata_available`` reflects whether fruit_database.json actually
    #    loaded valid entries — used by /health to avoid reporting "healthy"
    #    while a required production dependency is missing.
    if not metadata_available:
        logger.warning("Fruit metadata is unavailable; shelf-life will report 'unsupported'.")

    # 3. Load DetectionPipeline (YOLO)
    try:
        pipeline = DetectionPipeline(
            config=DetectionPipelineConfig(
                shelf_life_enabled=shelf_life_config.enabled,
                default_storage_condition=shelf_life_config.default_storage_condition,
            ),
            predictor=predictor
        )
        pipeline.initialize()
        logger.info("DetectionPipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize DetectionPipeline: {e}")
        pipeline = None
        
    yield
    
    # Shutdown
    if pipeline:
        pipeline.shutdown()
        logger.info("DetectionPipeline shut down.")

app = FastAPI(
    title="SmartFreshAI Inference API",
    description="Production API for fruit freshness and shelf-life estimation.",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check API health and inference readiness."""
    global pipeline, predictor, metadata_available
    
    detector_loaded = pipeline is not None and pipeline.detector is not None
    classifier_loaded = predictor is not None
    inference_ready = detector_loaded  # At minimum, detection must work
    
    device = "unknown"
    if pipeline and pipeline.detector:
        device = str(getattr(pipeline.detector, "device", "cpu"))
    
    status = "ok" if inference_ready else "error"
    
    return HealthResponse(
        status=status,
        inference_ready=inference_ready,
        detector_loaded=detector_loaded,
        classifier_loaded=classifier_loaded,
        metadata_available=metadata_available,
        device=device
    )

@app.post("/api/v1/detection/preview", response_model=DetectionPreviewResponse)
async def detect_preview(
    image: UploadFile = File(...),
):
    """Lightweight live-detection preview endpoint (camera scanning loop).

    This intentionally runs ONLY the YOLO detector — no freshness classification,
    no stabilization, no shelf-life estimation. The live scanner calls this a few
    times per second while the user holds a fruit in the frame (and only during
    the camera preview phase). Once a stable detection is confirmed, the scanner
    stops the loop and submits exactly ONE frozen frame to the full
    ``/api/v1/inference/image`` pipeline.

    Keeping this endpoint separate from the full-analysis pipeline means the
    camera UX stays responsive without re-executing efficientnet + shelf-life
    heuristics for every preview frame.
    """
    global pipeline

    if not pipeline or pipeline.detector is None:
        raise HTTPException(
            status_code=503, detail="Detection pipeline is not ready or failed to initialize."
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return DetectionPreviewResponse(
            success=False,
            message="Invalid or corrupted image format.",
            image=None,
            latency_ms=0.0,
            detections=[],
        )

    h, w = frame.shape[:2]
    if h < 32 or w < 32:
        return DetectionPreviewResponse(
            success=False,
            message="Image is too small for detection.",
            image=ImageMetaSchema(width=w, height=h),
            latency_ms=0.0,
            detections=[],
        )

    start_time = time.perf_counter()
    try:
        det_result = pipeline.detector.detect(frame)
    except Exception as e:
        logger.exception("Live detection failed.")
        raise HTTPException(status_code=500, detail=f"Live detection failed: {str(e)}")

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    detections = []
    for det in det_result.detections:
        detections.append(
            DetectedObjectPreviewSchema(
                label=det.label,
                class_id=det.class_id,
                confidence=round(float(det.confidence), 4),
                bbox=BoundingBoxSchema(
                    x1=det.bbox.x1, y1=det.bbox.y1, x2=det.bbox.x2, y2=det.bbox.y2
                ),
            )
        )

    # Only advertise the highest-confidence raw detections; the frontend
    # stability gate applies its own configurable threshold on top of this.
    detections.sort(key=lambda d: d.confidence, reverse=True)

    if detections:
        msg = f"Detected {len(detections)} fruit object(s)."
    else:
        msg = "No fruit detected."

    return DetectionPreviewResponse(
        success=True,
        message=msg,
        image=ImageMetaSchema(width=w, height=h),
        latency_ms=round(latency_ms, 2),
        detections=detections,
    )


@app.post("/api/v1/inference/image", response_model=InferenceResponse)
async def analyze_image(
    image: UploadFile = File(...),
    storage_condition: str = Form(None, description="Optional storage condition (e.g., ambient, refrigerated)")
):
    """Analyze an image to detect fruits, grade freshness, and estimate shelf life."""
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Inference pipeline is not ready or failed to initialize.")
        
    # Read image
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
        
    # Decode image using cv2
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image format. Please upload a valid image file.")
        
    # Check dimensions
    h, w = frame.shape[:2]
    if h < 32 or w < 32:
        raise HTTPException(status_code=400, detail="Image is too small (minimum 32x32).")
        
    # Storage condition is validated strictly and passed through
    # API -> DetectionPipeline -> ShelfLifeEstimator. Arbitrary text is
    # rejected with HTTP 400 instead of being silently ignored.
    if storage_condition is not None:
        try:
            storage_condition = normalize_storage_condition(storage_condition)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        
    # Process frame
    start_time = time.perf_counter()
    try:
        result = pipeline.process_frame(frame, storage_condition=storage_condition)
    except ValueError as exc:
        # Defensive: the pipeline re-validates before any model work.
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        logger.exception("Inference failed during process_frame.")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    
    # Serialize results
    try:
        fruits_data = []
        for fruit in result.fruits:
            # FruitResult.to_dict() returns all we need, just cast to Pydantic for validation
            fruit_dict = fruit.to_dict()
            fruits_data.append(FruitResultSchema(**fruit_dict))
            
        success = True
        msg = "Fruits detected successfully." if fruits_data else "No supported fruit detected."
        
        return InferenceResponse(
            success=success,
            message=msg,
            image=ImageMetaSchema(width=w, height=h),
            processing=ProcessingMetaSchema(
                latency_ms=latency_ms,
                unidentified_count=result.unidentified_count,
                fruit_count=len(fruits_data),
            ),
            fruits=fruits_data
        )
    except Exception as e:
        logger.exception("Failed to serialize inference results.")
        raise HTTPException(status_code=500, detail=f"Failed to serialize results: {str(e)}")
