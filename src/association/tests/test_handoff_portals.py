from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.association import AssociationConfig, Camera, Observation
from src.association.fragments import FragmentWindow
from src.association.geometry_tracking import _geometry
from src.association.handoff_portals import portal_evidence, validate_portals


def configuration():
    return [
        {
            "id": "door",
            "source_camera": "a",
            "target_camera": "b",
            "source_polygon": [[0, 0.6], [0.25, 0.6], [0.25, 0.97], [0, 0.97]],
            "target_polygon": [[0.4, 0.1], [0.65, 0.1], [0.65, 0.3], [0.4, 0.3]],
            "source_direction": [-1, 0],
            "target_direction": [0, 1],
            "expected_gap_s": 1.4,
            "timing_sigma_s": 1.2,
            "max_gap_s": 5,
            "reliability": 0.7,
        }
    ]


def window(fid, camera, start, points):
    return FragmentWindow(
        fid,
        camera,
        tuple(
            Observation(
                camera,
                i,
                start + i * 0.25,
                fid,
                (x * 1000 - 20, y * 1000 - 150, x * 1000 + 20, y * 1000),
                0.9,
            )
            for i, (x, y) in enumerate(points)
        ),
    )


def example():
    a = window("one", "a", 0, [(0.4 - i * 0.064, 0.85) for i in range(6)])
    b = window("two", "b", 2, [(0.52, 0.14 + i * 0.018) for i in range(6)])
    e = SimpleNamespace(
        scene_project={"handoff_portals": configuration()},
        cameras={c: Camera(c, width=1000, height=1000) for c in ["a", "b"]},
        floor=None,
        config=AssociationConfig(
            decision_mode="geometry_ambiguity",
            trajectory_handoffs=True,
            tracklet_threshold=0.62,
        ),
        boundary_geometry=None,
    )
    return e, a, b


def test_transfer_uses_both_gates_motion_and_time_without_inventing_floor_distance():
    e, a, b = example()
    before = deepcopy(e.scene_project)
    evidence, details = portal_evidence(e, a, b)
    assert evidence is not None and details["metric_distance_used"] is False
    score, decision = _geometry(e, a, b)
    assert score > 0.62 + 0.03 and decision["reason"] == "handoff_trajectory_portal"
    assert "calibration_validation" not in decision
    assert e.scene_project == before


@pytest.mark.parametrize(
    "case",
    [
        "wrong_source_zone",
        "wrong_target_zone",
        "wrong_direction",
        "stationary",
        "too_late",
        "long_overlap",
        "one_frame",
        "disabled",
    ],
)
def test_portal_cannot_merge_unqualified_candidates(case):
    e, a, b = example()
    if case == "wrong_source_zone":
        a = window("one", "a", 0, [(0.9 - i * 0.03, 0.85) for i in range(6)])
    if case == "wrong_target_zone":
        b = window("two", "b", 2, [(0.8, 0.14 + i * 0.018) for i in range(6)])
    if case == "wrong_direction":
        b = window("two", "b", 2, [(0.52, 0.25 - i * 0.018) for i in range(6)])
    if case == "stationary":
        b = window("two", "b", 2, [(0.52, 0.2)] * 6)
    if case == "too_late":
        b = replace(
            b,
            observations=tuple(
                replace(o, timestamp_s=o.timestamp_s + 10) for o in b.observations
            ),
        )
    if case == "long_overlap":
        b = replace(
            b,
            observations=tuple(
                replace(o, timestamp_s=o.timestamp_s - 2) for o in b.observations
            ),
        )
    if case == "one_frame":
        b = replace(b, observations=b.observations[:1])
    if case == "disabled":
        e.scene_project["handoff_portals"][0]["enabled"] = False
    assert portal_evidence(e, a, b) == (None, None)


def test_portal_configuration_validation():
    portals = configuration()
    assert validate_portals(portals, {"a", "b"}) == portals
    for key, value in [
        ("target_camera", "a"),
        ("source_direction", [0, 0]),
        ("max_gap_s", -1),
        ("source_polygon", [[0, 0], [2, 0], [2, 1]]),
    ]:
        invalid = deepcopy(portals)
        invalid[0][key] = value
        with pytest.raises((ValueError, TypeError)):
            validate_portals(invalid, {"a", "b"})


def test_explicit_portal_is_not_replaced_by_inaccurate_floor_gate_fallback():
    e, a, b = example()
    a = replace(
        a,
        observations=tuple(
            replace(o, detection_confidence=0.3) for o in a.observations
        ),
    )
    b = replace(
        b,
        observations=tuple(
            replace(o, detection_confidence=0.3) for o in b.observations
        ),
    )
    old = SimpleNamespace(
        available=True,
        value=0.01,
        reliability=0.95,
        discrepancy=3,
        reason="route:physically_too_fast",
        support_count=2,
        as_dict=lambda: {"reason": "route:physically_too_fast"},
    )
    e.boundary_geometry = SimpleNamespace(compare=lambda *args: old)
    score, detail = _geometry(e, a, b)
    assert detail["reason"] == "handoff_trajectory_portal" and score > 0.62


def test_portal_can_require_longer_motion_before_boosting_short_fragment():
    e, a, b = example()
    e.scene_project["handoff_portals"][0]["min_motion_span_s"] = 0.6
    short = window("two", "b", 2, [(0.52, 0.14 + i * 0.018) for i in range(3)])
    assert portal_evidence(e, a, short) == (None, None)
    evidence, detail = portal_evidence(e, a, b)
    assert evidence is not None and detail["min_motion_span_s"] == 0.6
    for value in [0.1, 2, True]:
        invalid = deepcopy(e.scene_project["handoff_portals"])
        invalid[0]["min_motion_span_s"] = value
        with pytest.raises(ValueError):
            validate_portals(invalid, {"a", "b"})
