"""Explicit typed camera graph: overlap is symmetric, transitions are directed."""

from math import hypot


class CameraGraph:
    def __init__(self, cameras):
        self.overlaps = {cid: set() for cid in cameras}
        self.transitions = {cid: set(c.transition_to) for cid, c in cameras.items()}
        for cid, camera in cameras.items():
            for other in camera.overlap_with:
                self.overlaps[cid].add(other)
                self.overlaps[other].add(cid)

    def relation(self, source, target, gap_s):
        if source == target:
            return "same_camera" if gap_s > 0 else None
        if target in self.overlaps[source]:
            return "overlap"
        if gap_s > 0 and target in self.transitions[source]:
            return "transition"
        # Reachability alone is insufficient evidence: never infer an overlap or teleport.
        return None

    @staticmethod
    def bonus(relation, value):
        return (
            value
            if relation == "overlap"
            else value * 0.5
            if relation == "transition"
            else 0.0
        )


def plausible_recovery(previous, current, gap_s):
    """Broad image-space motion gate, in person heights; no metric calibration implied."""
    ax1, ay1, ax2, ay2 = previous
    bx1, by1, bx2, by2 = current
    scale = max(ay2 - ay1, by2 - by1, 1)
    distance = hypot((ax1 + ax2 - bx1 - bx2) / 2, ay2 - by2) / scale
    return distance <= 1.5 + min(max(gap_s, 0), 4) * 0.5
