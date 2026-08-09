"""FreshSense AI - Inference modules.

Phase 1:
- predict.py: Single-image prediction

Phase 2:
- camera.py: Real-time webcam capture
- transforms.py: Deterministic inference preprocessing (no augmentations)
- fps.py: FPS monitoring and performance tracking
- overlay.py: Professional overlay rendering
- tracker.py: Prediction tracking with temporal smoothing
- predictor.py: Real-time prediction engine
- pipeline.py: Main inference pipeline orchestrator

Phase 3:
- stabilizer.py: Temporal prediction stabilization (EMA, majority voting, locking)
- quality.py: Image quality assessment (brightness, contrast, blur, motion)
- statistics.py: Session statistics and CSV/JSON logging
"""
