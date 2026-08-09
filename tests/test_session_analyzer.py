"""Tests for Phase 4 session analyzer module."""
import pytest

from src.inference.session_analyzer import SessionAnalyzer, SessionSummary


class TestSessionSummary:
    """Tests for SessionSummary dataclass."""

    def test_default_values(self):
        """Test default session summary values."""
        summary = SessionSummary(
            session_id="test",
            duration_seconds=10.0,
            total_frames=5,
            fruits_detected=3,
            unique_tracks=2,
        )
        assert summary.session_id == "test"
        assert summary.fresh_count == 0
        assert summary.avg_confidence == 0.0

    def test_custom_values(self):
        """Test custom session summary values."""
        summary = SessionSummary(
            session_id="session_123",
            duration_seconds=30.5,
            total_frames=100,
            fruits_detected=25,
            unique_tracks=5,
            fresh_count=15,
            stale_count=8,
            avg_confidence=0.85,
        )
        assert summary.session_id == "session_123"
        assert summary.fresh_count == 15
        assert summary.avg_confidence == 0.85


class TestSessionAnalyzer:
    """Tests for SessionAnalyzer."""

    def test_empty_session(self):
        """Test analysis of empty session."""
        summary = SessionAnalyzer().analyze("empty")
        assert summary.total_frames == 0
        assert summary.fruits_detected == 0
        assert summary.unique_tracks == 0

    def test_single_frame(self):
        """Test analysis with a single frame."""
        analyzer = SessionAnalyzer()
        analyzer.add_frame(
            {
                "timestamp": 1000.0,
                "predictions": [{"fruit_name": "apple", "freshness_class": "fresh"}],
                "confidences": [0.95],
                "tracking_ids": [1],
            }
        )
        summary = analyzer.analyze("test_1")
        assert summary.total_frames == 1
        assert summary.fruits_detected == 1
        assert summary.unique_tracks == 1
        assert summary.fresh_count == 1
        assert summary.avg_confidence == 0.95

    def test_class_switches(self):
        """Test detection of class switches in a track."""
        analyzer = SessionAnalyzer()
        for i, cls in enumerate(["fresh", "fresh", "stale", "stale", "rotten"]):
            analyzer.add_frame(
                {
                    "timestamp": 1000.0 + i,
                    "predictions": [{"fruit_name": "apple", "freshness_class": cls}],
                    "confidences": [0.8],
                    "tracking_ids": [1],
                }
            )
        summary = analyzer.analyze("test_3")
        assert summary.class_switches == 2

    def test_multiple_fruits_per_frame(self):
        """Test detection of multiple fruits in one frame."""
        analyzer = SessionAnalyzer()
        analyzer.add_frame(
            {
                "timestamp": 1000.0,
                "predictions": [
                    {"fruit_name": "apple", "freshness_class": "fresh"},
                    {"fruit_name": "banana", "freshness_class": "stale"},
                ],
                "confidences": [0.95, 0.85],
                "tracking_ids": [1, 2],
            }
        )
        summary = analyzer.analyze("test_4")
        assert summary.fruits_detected == 2
        assert summary.unique_tracks == 2

    def test_lowest_confidence_fruit(self):
        """Test identification of lowest-confidence fruit."""
        analyzer = SessionAnalyzer()
        analyzer.add_frame(
            {
                "timestamp": 1000.0,
                "predictions": [
                    {"fruit_name": "apple", "freshness_class": "fresh"},
                    {"fruit_name": "banana", "freshness_class": "rotten"},
                ],
                "confidences": [0.95, 0.45],
                "tracking_ids": [1, 2],
            }
        )
        summary = analyzer.analyze("test_5")
        assert summary.lowest_confidence_fruit is not None
        assert summary.lowest_confidence_fruit["confidence"] == 0.45

    def test_text_report_generation(self):
        """Test text report generation."""
        analyzer = SessionAnalyzer()
        analyzer.add_frame(
            {
                "timestamp": 1000.0,
                "predictions": [{"fruit_name": "apple", "freshness_class": "fresh"}],
                "confidences": [0.95],
                "tracking_ids": [1],
            }
        )
        summary = analyzer.analyze("test_7")
        report = analyzer.to_text_report(summary)
        assert "SESSION ANALYSIS REPORT" in report
        assert "test_7" in report
        assert "FRUIT DETECTION" in report


class TestUniqueTrackSemantics:
    """Tests that prove unique-track counting is correct.

    Core guarantee: one physical fruit tracked across N frames
    contributes exactly 1 to unique_tracks, not N.
    """

    def test_one_fruit_90_frames_is_one_track(self):
        """Canonical test: banana tracked across 90 frames → 1 unique track."""
        analyzer = SessionAnalyzer()
        for i in range(90):
            analyzer.add_frame(
                {
                    "timestamp": 1000.0 + i,
                    "predictions": [{"fruit_name": "banana", "freshness_class": "fresh"}],
                    "confidences": [0.9],
                    "tracking_ids": [5],  # same tracking ID throughout
                }
            )
        summary = analyzer.analyze("single_fruit_90_frames")
        assert summary.unique_tracks == 1, (
            f"Expected 1 unique track, got {summary.unique_tracks}"
        )
        assert summary.fruits_detected == 1, (
            f"Expected fruits_detected=1, got {summary.fruits_detected}"
        )
        assert summary.total_detections == 90, (
            f"Expected 90 total_detections, got {summary.total_detections}"
        )
        assert summary.total_frames == 90

    def test_zero_fruits(self):
        """Session with no detections."""
        analyzer = SessionAnalyzer()
        for i in range(5):
            analyzer.add_frame(
                {
                    "timestamp": float(i),
                    "predictions": [],
                    "confidences": [],
                    "tracking_ids": [],
                }
            )
        summary = analyzer.analyze("zero_fruits")
        assert summary.unique_tracks == 0
        assert summary.fruits_detected == 0
        assert summary.total_detections == 0
        assert summary.total_frames == 5
        assert summary.frames_with_detections == 0

    def test_one_fruit_one_frame(self):
        """Single detection in a single frame."""
        analyzer = SessionAnalyzer()
        analyzer.add_frame(
            {
                "timestamp": 1000.0,
                "predictions": [{"fruit_name": "apple", "freshness_class": "fresh"}],
                "confidences": [0.95],
                "tracking_ids": [1],
            }
        )
        summary = analyzer.analyze("one_one")
        assert summary.unique_tracks == 1
        assert summary.fruits_detected == 1
        assert summary.total_detections == 1

    def test_multiple_simultaneous_tracks(self):
        """Multiple fruits visible at the same time."""
        analyzer = SessionAnalyzer()
        for i in range(10):
            analyzer.add_frame(
                {
                    "timestamp": float(i),
                    "predictions": [
                        {"fruit_name": "apple", "freshness_class": "fresh"},
                        {"fruit_name": "orange", "freshness_class": "stale"},
                        {"fruit_name": "banana", "freshness_class": "rotten"},
                    ],
                    "confidences": [0.9, 0.8, 0.7],
                    "tracking_ids": [1, 2, 3],
                }
            )
        summary = analyzer.analyze("three_fruits_10_frames")
        assert summary.unique_tracks == 3
        assert summary.fruits_detected == 3
        assert summary.total_detections == 30  # 3 per frame × 10 frames

    def test_fruit_disappears_and_reappears_same_id(self):
        """Fruit disappears for a few frames, reappears with same tracking ID."""
        analyzer = SessionAnalyzer()
        # Frames 0-2: track 7 visible
        for i in range(3):
            analyzer.add_frame(
                {
                    "timestamp": float(i),
                    "predictions": [{"fruit_name": "apple", "freshness_class": "fresh"}],
                    "confidences": [0.9],
                    "tracking_ids": [7],
                }
            )
        # Frames 3-4: no detections
        for i in range(3, 5):
            analyzer.add_frame(
                {
                    "timestamp": float(i),
                    "predictions": [],
                    "confidences": [],
                    "tracking_ids": [],
                }
            )
        # Frames 5-7: track 7 reappears
        for i in range(5, 8):
            analyzer.add_frame(
                {
                    "timestamp": float(i),
                    "predictions": [{"fruit_name": "apple", "freshness_class": "fresh"}],
                    "confidences": [0.85],
                    "tracking_ids": [7],  # same ID
                }
            )
        summary = analyzer.analyze("disappear_reappear")
        assert summary.unique_tracks == 1, "Same ID across gap should be 1 track"
        assert summary.fruits_detected == 1
        assert summary.total_detections == 6  # 3 + 3

    def test_fruit_disappears_new_id_on_return(self):
        """Fruit disappears and gets a NEW tracking ID on return → 2 tracks."""
        analyzer = SessionAnalyzer()
        for i in range(3):
            analyzer.add_frame(
                {
                    "timestamp": float(i),
                    "predictions": [{"fruit_name": "apple", "freshness_class": "fresh"}],
                    "confidences": [0.9],
                    "tracking_ids": [10],
                }
            )
        for i in range(3, 6):
            analyzer.add_frame(
                {
                    "timestamp": float(i),
                    "predictions": [{"fruit_name": "apple", "freshness_class": "fresh"}],
                    "confidences": [0.85],
                    "tracking_ids": [11],  # NEW ID — tracker lost the fruit
                }
            )
        summary = analyzer.analyze("new_id_on_return")
        assert summary.unique_tracks == 2
        assert summary.fruits_detected == 2

    def test_class_switches_per_track(self):
        """Class switches are counted per track, not per frame."""
        analyzer = SessionAnalyzer()
        sequence = ["fresh", "fresh", "stale", "stale", "rotten"]
        for i, cls in enumerate(sequence):
            analyzer.add_frame(
                {
                    "timestamp": float(i),
                    "predictions": [{"fruit_name": "apple", "freshness_class": cls}],
                    "confidences": [0.8],
                    "tracking_ids": [1],
                }
            )
        summary = analyzer.analyze("class_switches")
        # fresh→stale (1) + stale→rotten (1) = 2 switches
        assert summary.class_switches == 2

    def test_frames_with_detections_counted_correctly(self):
        """frames_with_detections counts frames that had ≥1 fruit."""
        analyzer = SessionAnalyzer()
        for i in range(10):
            if i % 2 == 0:  # even frames: have detection
                analyzer.add_frame(
                    {
                        "timestamp": float(i),
                        "predictions": [{"fruit_name": "apple", "freshness_class": "fresh"}],
                        "confidences": [0.9],
                        "tracking_ids": [1],
                    }
                )
            else:           # odd frames: empty
                analyzer.add_frame(
                    {
                        "timestamp": float(i),
                        "predictions": [],
                        "confidences": [],
                        "tracking_ids": [],
                    }
                )
        summary = analyzer.analyze("alternating")
        assert summary.total_frames == 10
        assert summary.frames_with_detections == 5