"""Soft floor-space evidence; geometry never replaces appearance or camera topology."""

from bisect import bisect_left
from collections import Counter
from math import exp, hypot, isfinite
from statistics import median


class FloorMatcher:
    def __init__(self, project, cameras):
        self.project = project
        self.cameras = {c.camera_id: c for c in cameras}
        self.calibrations = {
            c["id"]: c["calibration"]
            for c in project["cameras"]
            if c.get("calibration")
        }
        room = project["room"]
        self.width, self.height = room["width_m"], room["height_m"]
        # Relative plans have no physical speed units. Normalize by plan diagonal.
        self.scale = (
            1.0
            if project.get("coordinate_units", "meters") == "meters"
            else hypot(self.width, self.height) / 25
        )
        self.radius = max(
            0.15 * self.scale,
            min(
                project.get("processing", {}).get(
                    "fusion_distance_m", 0.8 * self.scale
                ),
                1.2 * self.scale,
            ),
        )
        self.speed = 2.0 * self.scale
        self.cache = {}
        self.stats = Counter()

    def point(self, observation):
        key = observation.reference
        if key in self.cache:
            return self.cache[key]
        result = None
        calibration = self.calibrations.get(observation.camera_id)
        cam = self.cameras[observation.camera_id]
        x1, _, x2, y = observation.bbox_xyxy
        # Cropped feet and the other side of a projective horizon are unreliable.
        if calibration and (cam.height is None or y < cam.height * 0.985):
            h = calibration["matrix"]
            x = (x1 + x2) / 2
            d = h[2][0] * x + h[2][1] * y + h[2][2]
            anchors = calibration.get("video_points", [])
            anchor_d = (
                median([h[2][0] * a + h[2][1] * b + h[2][2] for a, b in anchors])
                if anchors
                else d
            )
            if abs(d) > 1e-8 and d * anchor_d > 0:
                px = (h[0][0] * x + h[0][1] * y + h[0][2]) / d
                py = (h[1][0] * x + h[1][1] * y + h[1][2]) / d
                if (
                    isfinite(px)
                    and isfinite(py)
                    and 0 <= px <= self.width
                    and 0 <= py <= self.height
                ):
                    result = (px, py)
        self.cache[key] = result
        self.stats["projected" if result is not None else "unavailable"] += 1
        return result

    def samples(self, observations):
        return [
            (o.timestamp_s, *p)
            for o in observations
            if (p := self.point(o)) is not None
        ]

    def predict(self, history, time, point):
        """Expanding ellipse in the velocity direction, fading over long gaps."""
        if not history:
            return None
        end = max(p[0] for p in history)
        dt = time - end
        if dt < -1e-6 or dt > 4:
            return None
        tail = [p for p in history if end - p[0] <= 0.25]
        x, y = median(p[1] for p in tail), median(p[2] for p in tail)
        older = [p for p in history if 0.4 <= end - p[0] <= 1.4]
        vx = vy = 0.0
        if older:
            ot = median(p[0] for p in older)
            vx = (x - median(p[1] for p in older)) / (end - ot)
            vy = (y - median(p[2] for p in older)) / (end - ot)
        velocity = hypot(vx, vy)
        if velocity > self.speed:
            vx *= self.speed / velocity
            vy *= self.speed / velocity
            velocity = self.speed
        if velocity < self.radius * 0.5:
            vx = vy = velocity = 0.0
        # Prediction beyond two seconds adds uncertainty rather than extrapolating forever.
        ahead = min(dt, 2)
        dx, dy = point[0] - x - vx * ahead, point[1] - y - vy * ahead
        across = self.radius + self.speed * dt * 0.45
        along = across + velocity * dt * 0.6
        if velocity:
            u = (dx * vx + dy * vy) / velocity
            v = (dx * -vy + dy * vx) / velocity
        else:
            u, v = dx, dy
        distance = hypot(u / along, v / across)
        return self.adjustment(distance) * exp(-dt / 4)

    @staticmethod
    def adjustment(distance):
        # Close support boosts weak appearance; contradiction softly penalizes it.
        return 0.12 * exp(-0.5 * distance * distance) - 0.20 * min(
            1, max(0, (distance - 1.5) / 2)
        )

    def online(self, observation, history):
        point = self.point(observation)
        if point is None:
            return None
        return self.predict(history, observation.timestamp_s, point)

    def between(self, left, right):
        """Sample simultaneous overlaps, otherwise predict across segment boundaries."""
        a, b = sorted(self.samples(left)), sorted(self.samples(right))
        if not a or not b:
            return None
        values = []
        times = [p[0] for p in b]
        # Bound offline pair cost; use median to tolerate occlusion / bad box feet.
        step = max(1, len(a) // 32)
        for t, x, y in a[::step]:
            idx = bisect_left(times, t)
            choices = b[max(0, idx - 1) : idx + 1]
            if not choices:
                continue
            q = min(choices, key=lambda p: abs(p[0] - t))
            if abs(q[0] - t) <= 0.25:
                distance = hypot(q[1] - x, q[2] - y) / (
                    self.radius + self.speed * abs(q[0] - t)
                )
                values.append(self.adjustment(distance))
        if values:
            return median(values)
        earlier, later = (a, b) if a[-1][0] < b[0][0] else (b, a)
        if later[0][0] < earlier[-1][0]:
            return None
        t, x, y = later[0]
        return self.predict(earlier, t, (x, y))
