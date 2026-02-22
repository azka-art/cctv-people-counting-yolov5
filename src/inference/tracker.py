"""
Simple IoU-based multi-object tracker (SORT-lite).

Lightweight tracker that assigns persistent IDs to detected people across
video frames using IoU (Intersection over Union) matching. This enables:
- Unique person counting (not just per-frame detection count)
- Track visualization with colored IDs
- Entry/exit counting potential (with tripwire extension)

Based on the SORT algorithm (Bewley et al., 2016) but simplified:
- No Kalman filter prediction (uses last-known position directly)
- IoU-only association (no appearance features)
- Suitable for static/slow-moving CCTV scenarios typical of BRT stations

Usage:
    tracker = SimpleTracker(max_disappeared=30, iou_threshold=0.3)
    for frame in video:
        detections = detect_people(model, frame)
        tracks = tracker.update(detections)
        # tracks: list of {x1, y1, x2, y2, score, track_id}
"""

import numpy as np
from collections import OrderedDict


class SimpleTracker:
    """
    IoU-based multi-object tracker for persistent person identification.

    Assigns unique integer IDs to detections and maintains them across frames
    using IoU matching between consecutive frames.

    Args:
        max_disappeared: Number of consecutive frames an object can be missing
                         before its track is deleted (default: 30 = ~1 sec at 30fps)
        iou_threshold: Minimum IoU to consider a detection as matching an
                       existing track (default: 0.3)
    """

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.3):
        self.next_id = 0
        self.tracks = OrderedDict()       # track_id -> {box, score, disappeared}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.total_unique = 0              # Total unique IDs assigned

    def update(self, detections: list) -> list:
        """
        Update tracks with new frame detections.

        Args:
            detections: list of dicts with keys x1, y1, x2, y2, score

        Returns:
            list of dicts with keys x1, y1, x2, y2, score, track_id
        """
        # If no detections, mark all existing tracks as disappeared
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["disappeared"] += 1
                if self.tracks[track_id]["disappeared"] > self.max_disappeared:
                    del self.tracks[track_id]
            return []

        # If no existing tracks, register all detections as new
        if len(self.tracks) == 0:
            results = []
            for det in detections:
                track_id = self._register(det)
                results.append({**det, "track_id": track_id})
            return results

        # Compute IoU between all existing tracks and new detections
        track_ids = list(self.tracks.keys())
        track_boxes = np.array([
            [self.tracks[tid]["box"]["x1"], self.tracks[tid]["box"]["y1"],
             self.tracks[tid]["box"]["x2"], self.tracks[tid]["box"]["y2"]]
            for tid in track_ids
        ])
        det_boxes = np.array([
            [d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections
        ])

        iou_matrix = self._compute_iou_matrix(track_boxes, det_boxes)

        # Greedy matching: highest IoU first
        matched_tracks = set()
        matched_dets = set()
        results = []

        # Sort all IoU pairs descending
        if iou_matrix.size > 0:
            pairs = []
            for i in range(iou_matrix.shape[0]):
                for j in range(iou_matrix.shape[1]):
                    if iou_matrix[i, j] >= self.iou_threshold:
                        pairs.append((iou_matrix[i, j], i, j))
            pairs.sort(reverse=True)

            for _, track_idx, det_idx in pairs:
                if track_idx in matched_tracks or det_idx in matched_dets:
                    continue
                matched_tracks.add(track_idx)
                matched_dets.add(det_idx)

                tid = track_ids[track_idx]
                det = detections[det_idx]
                self.tracks[tid]["box"] = det
                self.tracks[tid]["score"] = det["score"]
                self.tracks[tid]["disappeared"] = 0
                results.append({**det, "track_id": tid})

        # Handle unmatched tracks (disappeared)
        for i, tid in enumerate(track_ids):
            if i not in matched_tracks:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]

        # Handle unmatched detections (new tracks)
        for j, det in enumerate(detections):
            if j not in matched_dets:
                track_id = self._register(det)
                results.append({**det, "track_id": track_id})

        return results

    def _register(self, detection: dict) -> int:
        """Register a new track and return its ID."""
        track_id = self.next_id
        self.tracks[track_id] = {
            "box": detection,
            "score": detection["score"],
            "disappeared": 0,
        }
        self.next_id += 1
        self.total_unique += 1
        return track_id

    def get_unique_count(self) -> int:
        """Return total number of unique person IDs assigned since reset."""
        return self.total_unique

    def get_active_count(self) -> int:
        """Return number of currently active (visible) tracks."""
        return sum(1 for t in self.tracks.values() if t["disappeared"] == 0)

    def reset(self):
        """Reset tracker state."""
        self.next_id = 0
        self.tracks = OrderedDict()
        self.total_unique = 0

    @staticmethod
    def _compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """
        Compute IoU matrix between two sets of boxes.

        Args:
            boxes_a: (N, 4) array [x1, y1, x2, y2]
            boxes_b: (M, 4) array [x1, y1, x2, y2]

        Returns:
            (N, M) IoU matrix
        """
        n = boxes_a.shape[0]
        m = boxes_b.shape[0]
        iou = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                xa1 = max(boxes_a[i, 0], boxes_b[j, 0])
                ya1 = max(boxes_a[i, 1], boxes_b[j, 1])
                xa2 = min(boxes_a[i, 2], boxes_b[j, 2])
                ya2 = min(boxes_a[i, 3], boxes_b[j, 3])

                inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)

                area_a = (boxes_a[i, 2] - boxes_a[i, 0]) * (boxes_a[i, 3] - boxes_a[i, 1])
                area_b = (boxes_b[j, 2] - boxes_b[j, 0]) * (boxes_b[j, 3] - boxes_b[j, 1])

                union = area_a + area_b - inter
                iou[i, j] = inter / union if union > 0 else 0

        return iou