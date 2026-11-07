from copy import deepcopy
from dataclasses import replace

import pytest

from src.association import AssociationConfig, AssociationEngine, Camera, Observation
from src.association.scene_filter import (
    finish_scene_filter,
    select_floor_observations,
    validate_floor_filter,
)
from src.floor_mesh import calibrate_mesh

POLYGON = [[20, 20], [80, 20], [80, 80], [20, 80]]
CAM = Camera("a", width=100, height=100)


def obs(t=0, x=50, y=60, local="p", conf=0.9):
    return Observation("a", round(t * 100), t, local, (x - 5, y - 20, x + 5, y), conf)


def project(mode="mesh"):
    return {
        "cameras": [
            {
                "id": "a",
                "floor_filter": {"mode": mode, "allowed_polygon": POLYGON},
                "calibration": calibrate_mesh(POLYGON, POLYGON),
            }
        ]
    }


def select(rows, p=None):
    return select_floor_observations(rows, p or project(), {"a": CAM})


def test_bottom_center_not_box_center_and_boundary_is_inside():
    good, bad, _ = select([obs(x=20, y=80), obs(0.2, x=50, y=85, local="outside")])
    assert len(good) == len(bad) == 1
    assert bad[0][1] == "outside_floor_zone"


def test_grace_is_near_boundary_and_cannot_refresh_itself():
    rows = [
        obs(0, y=80),
        obs(0.5, y=82),
        obs(1, y=82),
        obs(1.1, y=82),
        obs(0.4, y=90, local="far"),
        obs(0.4, y=82, local="new"),
    ]
    good, bad, _ = select(rows)
    assert [o.timestamp_s for o in good] == [0, 0.5, 1]
    assert len(bad) == 3


def test_exclusions_have_no_grace_and_persist_with_mode_off():
    p = project("off")
    p["cameras"][0]["floor_filter"]["excluded_polygons"] = [
        [[40, 50], [60, 50], [60, 70], [40, 70]]
    ]
    good, bad, _ = select([obs(0, y=40), obs(0.2, y=60)], p)
    assert len(good) == 1 and bad[0][1] == "excluded_polygon"


def test_missing_calibration_fails_open_and_reports_camera():
    p = project()
    p["cameras"][0]["calibration"] = None
    good, bad, report = select([obs(x=99)], p)
    assert len(good) == 1 and not bad and report["unavailable_zone_cameras"] == ["a"]


def test_concave_polygon_does_not_fill_notch():
    p = project("polygon")
    p["cameras"][0]["floor_filter"]["allowed_polygon"] = [
        [0, 0],
        [90, 0],
        [90, 90],
        [60, 90],
        [60, 40],
        [40, 40],
        [40, 90],
        [0, 90],
    ]
    good, bad, _ = select([obs(x=50, y=60)], p)
    assert not good and len(bad) == 1


@pytest.mark.parametrize(
    "polygon",
    [
        [[0, 0], [90, 90], [0, 90], [90, 0]],
        [[0, 0], [1, 1], [2, 2]],
        [[0, 0], [1, 0], [float("nan"), 1]],
        [[0, 0], [1, 0], [1, 1], [0, 0]],
    ],
)
def test_bad_polygons_are_rejected(polygon):
    with pytest.raises(ValueError):
        validate_floor_filter({"mode": "polygon", "allowed_polygon": polygon})


def make_episode(gid, rows):
    a = [
        {
            "camera_id": o.camera_id,
            "frame_index": o.frame_index,
            "local_track_id": o.local_track_id,
            "timestamp_s": o.timestamp_s,
            "global_track_id": gid,
            "reason": "test",
        }
        for o in rows
    ]
    t = {
        "global_track_id": gid,
        "start_s": rows[0].timestamp_s,
        "last_seen_s": rows[-1].timestamp_s,
        "end_s": rows[-1].timestamp_s,
        "camera_ids": sorted({o.camera_id for o in rows}),
        "status": "closed",
        "close_reason": "end_of_video",
        "confirmation_state": "confirmed",
        "confirmed_at_s": rows[0].timestamp_s,
    }
    return a, t


def finish(episodes):
    rows = []
    a = []
    t = []
    for gid, rr in episodes:
        rows += rr
        aa, tt = make_episode(gid, rr)
        a += aa
        t.append(tt)
    return finish_scene_filter(rows, a, t, [], {})


def test_real_coherent_standing_person_is_retained_even_at_low_confidence():
    for confidence in [0.3, 0.95]:
        _, tracks, report = finish(
            [("g1", [obs(t, conf=confidence) for t in [0, 0.2, 0.4, 0.6, 1]])]
        )
        assert report["stationary_suppressed"] == 0
        assert tracks[0]["confirmation_state"] == "confirmed"


def test_repeated_weak_stationary_hotspot_is_suppressed():
    episodes = [
        (f"g{i}", [obs(start + d, local=str(i), conf=0.3) for d in [0, 0.2, 0.4, 0.6]])
        for i, start in enumerate([0, 4, 8])
    ]
    assignments, tracks, report = finish(episodes)
    assert report["stationary_suppressed"] == 3
    assert all(a["scene_state"] == "suppressed" for a in assignments)
    assert all(t["confirmation_state"] == "suppressed" for t in tracks)


def test_high_confidence_repeated_standing_people_are_not_suppressed():
    episodes = [
        (f"g{i}", [obs(start + d, local=str(i)) for d in [0, 0.2, 0.4]])
        for i, start in enumerate([0, 4, 8])
    ]
    assert finish(episodes)[2]["stationary_suppressed"] == 0


def test_low_confidence_static_flicker_suppressed_but_moving_flicker_kept():
    for moving in [False, True]:
        rows = [
            obs(t, x=50 + (t * 20 if moving else 0), conf=0.3)
            for t in [0, 0.2, 0.4, 1.4, 1.6]
        ]
        assert finish([("g1", rows)])[2]["stationary_suppressed"] == (
            0 if moving else 1
        )


def test_long_or_multicamera_episode_is_protected():
    for rows in [
        [obs(t, conf=0.3) for t in [0, 0.2, 0.4, 1.4, 5]],
        [obs(t, conf=0.3) for t in [0, 0.2, 0.4, 1.4]]
        + [replace(obs(1.6, conf=0.3), camera_id="b")],
    ]:
        assert finish([("g1", rows)])[2]["stationary_suppressed"] == 0


def test_engine_all_outside_preserves_every_reference_with_diagnostic_ids():
    rows = [obs(t, x=95) for t in [0, 0.2, 0.4]]
    p = project()
    p["room"] = {"width_m": 100, "height_m": 100}
    original = deepcopy(p)
    engine = AssociationEngine(
        [CAM],
        AssociationConfig(
            decision_mode="geometry_ambiguity", filter_scene=True, confirm_tracks=True
        ),
        p,
    )
    a, tracks = engine.run(rows)
    assert len(a) == len(rows)
    assert {(r["camera_id"], r["frame_index"], r["local_track_id"]) for r in a} == {
        o.reference for o in rows
    }
    assert len(tracks) == 1 and tracks[0]["scene_state"] == "suppressed"
    assert engine.confirmation_report["confirmed"] == 0
    assert engine.scene_report["rejected_observations"] == 3
    assert p == original
