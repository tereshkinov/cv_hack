from types import SimpleNamespace

import pytest

from src.association import AssociationConfig, AssociationEngine
from src.association.fragments import FragmentWindow
from src.association.handoff import joint_handoffs, solve_matching, trajectory_evidence

from .test_geolocation import obs, setup


def node(fid, camera, offset=0, reverse=False, start=0):
    rows = [
        obs(camera, fid, start + t, 100 + offset + (-40 * t if reverse else 40 * t))
        for t in [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
    ]
    return FragmentWindow(fid, camera, tuple(rows))


def engine():
    cams, p = setup()
    return AssociationEngine(
        cams,
        AssociationConfig(
            decision_mode="geometry_ambiguity",
            trajectory_handoffs=True,
            tracklet_threshold=0.62,
        ),
        p,
    )


def test_accumulates_head_support_and_does_not_use_first_frame_only():
    e = engine()
    a = node("a", "a")
    b = node("b", "b")
    short = FragmentWindow("s", "b", b.observations[:1])
    evidence, details = trajectory_evidence(e, a, short)
    assert evidence is None and details["reason"] == "handoff_pending_trajectory"
    evidence, details = trajectory_evidence(e, a, b)
    assert evidence is not None and details["left_samples"] >= 3
    assert details["common_span_s"] >= 0.4


def test_opposed_trajectories_score_lower_than_same_direction():
    e = engine()
    a = node("a", "a")
    same, _ = trajectory_evidence(e, a, node("b", "b", offset=1))
    opposite, details = trajectory_evidence(
        e, a, node("c", "b", offset=48, reverse=True)
    )
    assert same[0] > opposite[0]
    assert details["direction_cosine"] < -0.9


def test_stationary_people_have_no_arbitrary_direction_penalty():
    e = engine()
    a = FragmentWindow(
        "a", "a", tuple(obs("a", "one", t, 100) for t in [0, 0.2, 0.4, 0.6])
    )
    b = FragmentWindow(
        "b", "b", tuple(obs("b", "two", t, 101) for t in [0, 0.2, 0.4, 0.6])
    )
    evidence, details = trajectory_evidence(e, a, b)
    assert evidence[0] > 0.9 and details["direction_cosine"] is None


def test_exact_matching_beats_greedy_and_no_match_is_available():
    weights = {("a", "x"): 0.9, ("b", "x"): 0.89, ("a", "y"): 0.88, ("b", "y"): 0.1}
    score, links = solve_matching(["a", "b"], ["x", "y"], weights)
    assert score == pytest.approx(1.77)
    assert set(links) == {("a", "y"), ("b", "x")}
    score, links = solve_matching(["a"], ["x", "y"], {("a", "x"): -0.1, ("a", "y"): 0})
    assert score == 0 and not links


def test_equal_joint_alternatives_are_deferred():
    nodes = [node("a", "a"), node("b", "a"), node("x", "b"), node("y", "b")]
    cache = {
        (a, b): (0.8, {"handoff": {"reason": "handoff_trajectory_overlap"}})
        for a in ["a", "b"]
        for b in ["x", "y"]
    }
    decisions, packets = joint_handoffs(nodes, cache, 0.62, 0.03)
    assert len(packets) == 1 and not any(d["selected"] for d in decisions.values())


def test_joint_assignment_certifies_consistent_pairs():
    nodes = [node("a", "a"), node("b", "a"), node("x", "b"), node("y", "b")]
    scores = {("a", "x"): 0.90, ("b", "x"): 0.89, ("a", "y"): 0.88, ("b", "y"): 0.70}
    cache = {
        key: (value, {"handoff": {"reason": "handoff_trajectory_overlap"}})
        for key, value in scores.items()
    }
    decisions, _ = joint_handoffs(nodes, cache, 0.62, 0.03)
    assert {key for key, d in decisions.items() if d["selected"]} == {
        ("a", "y"),
        ("b", "x"),
    }


def test_route_corner_uses_route_tangents_instead_of_straight_prediction():
    e = engine()
    a = node("a", "a")
    b = node("b", "b", offset=500, start=3)
    route = SimpleNamespace(
        available=True,
        reason="route:plausible",
        reliability=0.45,
        discrepancy=0.1,
        expected_region={
            "trajectory": [
                {"x_m": 0, "y_m": 0},
                {"x_m": 5, "y_m": 0},
                {"x_m": 5, "y_m": 5},
            ]
        },
    )
    evidence, details = trajectory_evidence(e, a, b, route)
    assert evidence is not None and details["route_reason"] == "route:plausible"
    assert "position_residual" not in details


def test_offline_engine_preserves_refs_and_shared_id_with_delayed_evidence():
    e = engine()
    rows = list(node("a", "a").observations + node("b", "b", offset=1).observations)
    assignments, tracks = e.run(rows)
    assert len(tracks) == 1
    assert len(assignments) == len(rows)
    assert e.fragment_report["handoff"]["trajectory_edges"] >= 1
    assert tracks[0]["start_s"] == 0


@pytest.mark.parametrize("start", [1.2, 1.0])
def test_touching_or_briefly_overlapping_views_use_tail_and_head(start):
    e = engine()
    a = node("first", "a")
    b = node("second", "b", offset=40 * start, start=start)
    evidence, detail = trajectory_evidence(e, a, b)
    assert evidence is not None
    assert detail["reason"] == "handoff_trajectory_boundary"
    assert detail["left_span_s"] >= 0.4 and detail["right_span_s"] >= 0.4
    assignments, tracks = e.run(a.observations + b.observations)
    assert len(tracks) == 1
    assert len(assignments) == len(a.observations + b.observations)
    assert e.fragment_report["edges"][0]["selected"]


def test_boundary_transfer_still_needs_a_real_head_trajectory():
    e = engine()
    a = node("first", "a")
    b = node("second", "b", offset=48, start=1.2)
    b = FragmentWindow(b.fragment_id, b.camera_id, b.observations[:1])
    assert trajectory_evidence(e, a, b)[1]["reason"] == "handoff_pending_trajectory"
    assert len(e.run(a.observations + b.observations)[1]) == 2


def test_far_boundary_positions_cannot_merge_even_with_equal_direction():
    e = engine()
    a = node("first", "a")
    b = node("second", "b", offset=500, start=1.2)
    assert len(e.run(a.observations + b.observations)[1]) == 2


def test_boundary_uncertainty_accounts_for_both_unchecked_camera_projections():
    from src.association.geometry_tracking import _geometry
    from src.association.models import Observation

    # Synthetic floor paths with a modest cross-view registration offset.
    # Each path is coherent; a shared-view duration does not exist at all.
    def window(fid, cam, times, origin, v):
        rows = tuple(
            Observation(
                cam,
                i,
                t,
                fid,
                (
                    origin[0] + v[0] * (t - times[0]) - 0.01,
                    origin[1] + v[1] * (t - times[0]) - 0.1,
                    origin[0] + v[0] * (t - times[0]) + 0.01,
                    origin[1] + v[1] * (t - times[0]),
                ),
                0.9,
            )
            for i, t in enumerate(times)
        )
        return FragmentWindow(fid, cam, rows)

    a = window("first", "a", [0, 0.2, 0.4, 0.6, 1.2, 1.4], (3.72, 2.44), (-0.35, -0.18))
    b = window(
        "second",
        "b",
        [1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8],
        (3.52, 2.35),
        (-0.1, -0.1),
    )
    floor = SimpleNamespace(
        projection=lambda o: SimpleNamespace(
            available=True,
            reliability=1,
            point=((o.bbox_xyxy[0] + o.bbox_xyxy[2]) / 2, o.bbox_xyxy[3]),
        ),
        radius=0.2,
        sync_uncertainty_s=0.25,
        sync_verified=False,
        metric_scale={"meters_per_unit": 4.21, "uncertainty_fraction": 0.15},
        calibration_checks={},
    )
    e = SimpleNamespace(
        floor=floor,
        config=AssociationConfig(
            decision_mode="geometry_ambiguity",
            trajectory_handoffs=True,
            tracklet_threshold=0.62,
        ),
        boundary_geometry=None,
    )
    score, detail = _geometry(e, a, b)
    assert score > 0.62 + 0.03
    assert detail["handoff"]["evidence_window"] == "tail_to_head"
    assert detail["direction_agreement_bonus"] > 0
