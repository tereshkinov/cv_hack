"""Diagnostic overlays for local tracking; never modify inference inputs."""
from collections import deque
import hashlib
import cv2


class PreviewOverlay:
    def __init__(self, trail_points=60, max_gap_s=0.75):
        self.trail_points = trail_points
        self.max_gap_s = max_gap_s
        self.history = {}
        self.seen = set()

    @staticmethod
    def color(identity):
        palette = [(0, 210, 255), (255, 180, 40), (90, 245, 80),
                   (225, 90, 255), (50, 140, 255), (230, 240, 70),
                   (180, 110, 255), (255, 210, 180)]
        index = int(identity[1:]) - 1 if identity.startswith('l') and identity[1:].isdigit() else hashlib.sha256(identity.encode()).digest()[0]
        return palette[index % len(palette)]

    def draw(self, frame, records, camera_id, frame_index, timestamp_s):
        height, width = frame.shape[:2]
        # Retain only recent trails. IDs ever seen are kept separately for the local counter.
        self.history = {key: points for key, points in self.history.items()
                        if timestamp_s - points[-1][0] <= 2.0}
        for identity, box, confidence in records:
            self.seen.add(identity)
            points = self.history.setdefault(identity, deque(maxlen=self.trail_points))
            x1, y1, x2, y2 = map(int, box)
            point = (max(0, min(width - 1, (x1 + x2) // 2)), max(0, min(height - 1, y2)))
            points.append((timestamp_s, point))
            color = self.color(identity)
            for (ta, pa), (tb, pb) in zip(points, list(points)[1:]):
                if tb - ta <= self.max_gap_s:
                    cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, point, 4, color, -1)
            text = f'{identity}  {confidence:.2f}'
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            text_y = min(height - baseline - 1, max(text_height + 4, y1 - 7))
            text_x = max(0, min(x1, width - text_width - 4))
            cv2.rectangle(frame, (text_x, text_y - text_height - 3),
                          (text_x + text_width + 3, text_y + baseline), (15, 15, 15), -1)
            cv2.putText(frame, text, (text_x + 1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)
        label = f'LOCAL IDs | {camera_id} | t={timestamp_s:.2f}s frame={frame_index} | visible={len(records)} seen={len(self.seen)}'
        cv2.rectangle(frame, (0, 0), (width, min(32, height)), (20, 20, 20), -1)
        cv2.putText(frame, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.25, min(0.52, width / 1400)), (255, 255, 255), 1, cv2.LINE_AA)
        return frame
