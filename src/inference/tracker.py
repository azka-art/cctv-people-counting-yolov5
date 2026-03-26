"""
SORT-based multi-object tracker with Kalman filter prediction.

Implements the SORT algorithm (Bewley et al., 2016) for persistent person
tracking across video frames. Upgrades from simple IoU-only matching to
Kalman-filter-predicted tracking, enabling:

- Persistent unique IDs across frames (even through brief occlusion)
- Kalman filter position prediction to bridge detection gaps
- ID switch counting for tracking quality evaluation
- Active / lost / total unique person statistics

Key difference from simple IoU tracker:
    IoU-only:   matches on LAST KNOWN position → loses tracks instantly on miss
    SORT (this): predicts NEXT position via Kalman → bridges 1–N frame gaps

Dependencies:
    pip install filterpy numpy

References:
    Bewley et al. (2016) "Simple Online and Realtime Tracking"
    https://arxiv.org/abs/1602.01783

Usage:
    tracker = SORTTracker(max_age=30, min_hits=1, iou_threshold=0.3)
    for frame in video:
        detections = detect_people(model, frame)
        tracks = tracker.update(detections)
        # tracks: list of {x1, y1, x2, y2, score, track_id}

    metrics = tracker.get_metrics()
    # {'total_unique': 42, 'id_switches': 3, 'active_tracks': 5, ...}
"""

import numpy as np
from collections import OrderedDict

try:
    from filterpy.kalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kalman-filtered single-object track
# ---------------------------------------------------------------------------

class KalmanTrack:
    """
    Single object track using a constant-velocity Kalman filter.

    State vector: [x_center, y_center, width, height, vx, vy, vw, vh]
    Observation:  [x_center, y_center, width, height]

    This formulation (from SORT paper) predicts where a bounding box will be
    in the next frame, allowing IoU matching even when detection is briefly
    absent.
    """

    _count = 0  # class-level ID counter (reset via SORTTracker.reset())

    def __init__(self, detection: dict):
        if not KALMAN_AVAILABLE:
            raise ImportError(
                "filterpy is required for Kalman tracking. "
                "Install with: pip install filterpy"
            )

        # Assign unique ID
        self.track_id = KalmanTrack._count
        KalmanTrack._count += 1

        self.hits = 1               # consecutive detection matches
        self.no_detection_count = 0 # consecutive frames without match
        self.score = detection["score"]

        # Build Kalman filter: dim_x=8 state, dim_z=4 observation
        kf = KalmanFilter(dim_x=8, dim_z=4)

        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ], dtype=float)

        # Observation matrix (we observe position and size, not velocity)
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=float)

        # Measurement noise — higher weight on position than velocity
        kf.R[2:, 2:] *= 10.0

        # Uncertainty in initial state — high for velocity components
        kf.P[4:, 4:] *= 1000.0
        kf.P *= 10.0

        # Process noise
        kf.Q[-1, -1] *= 0.01
        kf.Q[4:, 4:] *= 0.01

        # Initialize state from detection
        kf.x[:4] = self._box_to_z(detection)
        self.kf = kf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """Advance state estimate by one timestep. Returns predicted [x1,y1,x2,y2]."""
        # Clamp width/height to avoid negative values
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] = 0
        if self.kf.x[3] + self.kf.x[7] <= 0:
            self.kf.x[7] = 0
        self.kf.predict()
        self.no_detection_count += 1
        return self._z_to_box(self.kf.x)

    def update(self, detection: dict):
        """Update state with a matched detection."""
        self.kf.update(self._box_to_z(detection))
        self.score = detection["score"]
        self.hits += 1
        self.no_detection_count = 0

    def get_state(self) -> dict:
        """Return current estimated bounding box as a dict."""
        x1, y1, x2, y2 = self._z_to_box(self.kf.x).flatten()
        return {
            "x1": float(max(0, x1)),
            "y1": float(max(0, y1)),
            "x2": float(max(0, x2)),
            "y2": float(max(0, y2)),
            "score": float(self.score),
            "track_id": self.track_id,
        }

    # ------------------------------------------------------------------
    # Box ↔ Kalman state conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _box_to_z(det: dict) -> np.ndarray:
        """Convert {x1,y1,x2,y2} detection to Kalman observation [cx,cy,w,h]."""
        cx = (det["x1"] + det["x2"]) / 2.0
        cy = (det["y1"] + det["y2"]) / 2.0
        w  =  det["x2"] - det["x1"]
        h  =  det["y2"] - det["y1"]
        return np.array([[cx], [cy], [w], [h]], dtype=float)

    @staticmethod
    def _z_to_box(x: np.ndarray) -> np.ndarray:
        """Convert Kalman state [cx,cy,w,h,...] to [x1,y1,x2,y2]."""
        flat = x.flatten()
        cx, cy, w, h = float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ---------------------------------------------------------------------------
# Fallback: pure IoU tracker (no filterpy required)
# ---------------------------------------------------------------------------

class _IoUTrack:
    """
    Minimal IoU-only track (no Kalman). Used when filterpy is unavailable.
    Matches on last known position — no prediction across missed frames.
    """
    _count = 0

    def __init__(self, detection: dict):
        self.track_id = _IoUTrack._count
        _IoUTrack._count += 1
        self.box = detection
        self.score = detection["score"]
        self.no_detection_count = 0
        self.hits = 1

    def predict(self) -> np.ndarray:
        self.no_detection_count += 1
        return np.array([self.box["x1"], self.box["y1"],
                         self.box["x2"], self.box["y2"]])

    def update(self, detection: dict):
        self.box = detection
        self.score = detection["score"]
        self.no_detection_count = 0
        self.hits += 1

    def get_state(self) -> dict:
        return {**self.box, "score": self.score, "track_id": self.track_id}


# ---------------------------------------------------------------------------
# Main tracker — automatically selects Kalman or IoU-only
# ---------------------------------------------------------------------------

class SORTTracker:
    """
    SORT-based multi-object tracker with automatic Kalman/IoU fallback.

    Selects KalmanTrack if filterpy is installed, otherwise falls back
    to IoU-only tracking with a console warning.

    Args:
        max_age:        Max frames a track can be unmatched before deletion.
                        Higher = more tolerant of detection gaps (default: 30)
        min_hits:       Min consecutive hits before a track is returned as output.
                        Filters out spurious single-frame detections (default: 1)
        iou_threshold:  Min IoU for detection-to-track association (default: 0.3)

    Example:
        tracker = SORTTracker(max_age=30, min_hits=1, iou_threshold=0.3)
        tracks  = tracker.update(detections)   # per-frame call
        metrics = tracker.get_metrics()        # end-of-video summary
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self._tracks: list = []

        # Metrics
        self._total_unique: int = 0
        self._id_switches: int = 0
        self._prev_assignments: dict = {}   # det_index -> track_id (previous frame)
        self._frame_count: int = 0

        if not KALMAN_AVAILABLE:
            print(
                "[SORTTracker] WARNING: filterpy not installed. "
                "Falling back to IoU-only tracking (no Kalman prediction). "
                "Install with: pip install filterpy"
            )

    # ------------------------------------------------------------------
    # Core update loop
    # ------------------------------------------------------------------

    def update(self, detections: list) -> list:
        """
        Update tracker state with detections from current frame.

        Args:
            detections: List of dicts, each with keys: x1, y1, x2, y2, score

        Returns:
            List of dicts with keys: x1, y1, x2, y2, score, track_id
            Only returns tracks with >= min_hits consecutive matches.
        """
        self._frame_count += 1

        # --- Step 1: Predict new positions for all active tracks ---
        predicted_boxes = np.zeros((len(self._tracks), 4))
        tracks_to_delete = []
        for i, track in enumerate(self._tracks):
            pred = track.predict()
            predicted_boxes[i] = pred
            if np.any(np.isnan(pred)):
                tracks_to_delete.append(i)

        # Remove tracks with NaN predictions (degenerate state)
        for i in reversed(tracks_to_delete):
            self._tracks.pop(i)
            predicted_boxes = np.delete(predicted_boxes, i, axis=0)

        # --- Step 2: Associate detections to predicted track positions ---
        matched, unmatched_dets, unmatched_tracks = self._associate(
            detections, predicted_boxes
        )

        # --- Step 3: Update matched tracks ---
        current_assignments = {}
        for track_idx, det_idx in matched:
            self._tracks[track_idx].update(detections[det_idx])
            current_assignments[det_idx] = self._tracks[track_idx].track_id

        # --- Step 4: Count ID switches ---
        # An ID switch occurs when a detection that was previously matched
        # to track A is now matched to track B
        for det_idx, new_tid in current_assignments.items():
            if det_idx in self._prev_assignments:
                old_tid = self._prev_assignments[det_idx]
                if old_tid != new_tid:
                    self._id_switches += 1

        self._prev_assignments = current_assignments

        # --- Step 5: Create new tracks for unmatched detections ---
        for det_idx in unmatched_dets:
            track = self._create_track(detections[det_idx])
            self._tracks.append(track)
            self._total_unique += 1

        # --- Step 6: Delete old unmatched tracks ---
        self._tracks = [
            t for t in self._tracks
            if t.no_detection_count <= self.max_age
        ]

        # --- Step 7: Build output (only confirmed tracks) ---
        results = []
        for track in self._tracks:
            if track.hits >= self.min_hits or self._frame_count <= self.min_hits:
                results.append(track.get_state())

        return results

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict:
        """
        Return tracking quality metrics accumulated since last reset.

        Returns:
            dict with keys:
                total_unique  - Total unique person IDs assigned
                id_switches   - Number of ID switches detected
                active_tracks - Currently active tracks
                lost_tracks   - Tracks present but no recent detection
                frame_count   - Total frames processed
        """
        active = sum(1 for t in self._tracks if t.no_detection_count == 0)
        lost   = sum(1 for t in self._tracks if t.no_detection_count > 0)
        return {
            "total_unique":  self._total_unique,
            "id_switches":   self._id_switches,
            "active_tracks": active,
            "lost_tracks":   lost,
            "frame_count":   self._frame_count,
        }

    def get_unique_count(self) -> int:
        """Total unique person IDs assigned since last reset."""
        return self._total_unique

    def get_id_switches(self) -> int:
        """Total ID switches detected since last reset."""
        return self._id_switches

    def get_active_count(self) -> int:
        """Number of currently active (visible) tracks."""
        return sum(1 for t in self._tracks if t.no_detection_count == 0)

    def reset(self):
        """Reset all tracker state and metrics."""
        self._tracks = []
        self._total_unique = 0
        self._id_switches = 0
        self._prev_assignments = {}
        self._frame_count = 0
        # Reset class-level ID counters
        KalmanTrack._count = 0
        _IoUTrack._count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_track(self, detection: dict):
        """Create a new KalmanTrack or _IoUTrack depending on availability."""
        if KALMAN_AVAILABLE:
            return KalmanTrack(detection)
        return _IoUTrack(detection)

    def _associate(
        self,
        detections: list,
        predicted_boxes: np.ndarray,
    ) -> tuple:
        """
        Greedily associate detections to predicted track boxes via IoU.

        Returns:
            matched:          list of (track_idx, det_idx) pairs
            unmatched_dets:   set of detection indices with no match
            unmatched_tracks: set of track indices with no match
        """
        if len(self._tracks) == 0:
            return [], set(range(len(detections))), set()

        if len(detections) == 0:
            return [], set(), set(range(len(self._tracks)))

        det_boxes = np.array([
            [d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections
        ])
        iou_matrix = _compute_iou_matrix(predicted_boxes, det_boxes)

        # Greedy matching: highest IoU first
        matched = []
        matched_tracks = set()
        matched_dets = set()

        pairs = [
            (iou_matrix[i, j], i, j)
            for i in range(iou_matrix.shape[0])
            for j in range(iou_matrix.shape[1])
            if iou_matrix[i, j] >= self.iou_threshold
        ]
        pairs.sort(reverse=True)

        for _, t_idx, d_idx in pairs:
            if t_idx in matched_tracks or d_idx in matched_dets:
                continue
            matched.append((t_idx, d_idx))
            matched_tracks.add(t_idx)
            matched_dets.add(d_idx)

        unmatched_dets   = set(range(len(detections))) - matched_dets
        unmatched_tracks = set(range(len(self._tracks))) - matched_tracks

        return matched, unmatched_dets, unmatched_tracks


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

# Keep old name working so inference_video.py doesn't break immediately
SimpleTracker = SORTTracker


# ---------------------------------------------------------------------------
# Standalone IoU utility (used by evaluate.py too)
# ---------------------------------------------------------------------------

def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU matrix between two sets of boxes.

    Args:
        boxes_a: (N, 4) array [x1, y1, x2, y2]
        boxes_b: (M, 4) array [x1, y1, x2, y2]

    Returns:
        (N, M) float array of IoU values in [0, 1]
    """
    n, m = boxes_a.shape[0], boxes_b.shape[0]
    iou = np.zeros((n, m), dtype=float)

    for i in range(n):
        for j in range(m):
            xa1 = max(boxes_a[i, 0], boxes_b[j, 0])
            ya1 = max(boxes_a[i, 1], boxes_b[j, 1])
            xa2 = min(boxes_a[i, 2], boxes_b[j, 2])
            ya2 = min(boxes_a[i, 3], boxes_b[j, 3])

            inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
            area_a = (boxes_a[i, 2] - boxes_a[i, 0]) * (boxes_a[i, 3] - boxes_a[i, 1])
            area_b = (boxes_b[j, 2] - boxes_b[j, 0]) * (boxes_b[j, 3] - boxes_b[j, 1])
            union = area_a + area_b - inter
            iou[i, j] = inter / union if union > 0 else 0.0

    return iou
