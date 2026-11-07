from copy import deepcopy

from src.association.models import AssociationConfig
from src.association.short_stitch import stitch_short_tracks


def fixture():
    nodes = [
        {"fragment_id": fid, "start_s": start, "end_s": end}
        for fid, start, end in [("a", 0, 10), ("s", 11, 13), ("b", 14, 24)]
    ]
    tracks = [
        {
            "global_track_id": n["fragment_id"],
            "fragment_ids": [n["fragment_id"]],
            "start_s": n["start_s"],
            "last_seen_s": n["end_s"],
            "end_s": n["end_s"],
            "camera_ids": ["cam1"],
            "status": "closed",
            "close_reason": "end_of_video",
        }
        for n in nodes
    ]
    assignments = [
        {"global_track_id": n["fragment_id"], "frame_index": i}
        for i, n in enumerate(nodes)
    ]
    return (
        assignments,
        tracks,
        {
            "fragments": nodes,
            "edges": [
                {"left": "a", "right": "s", "score": 0.75, "geometry_quality": 0.8}
            ],
            "constraints": [],
        },
    )


def run(a, t, r):
    return stitch_short_tracks(
        a, t, r, AssociationConfig(tracklet_threshold=0.62, ambiguity_margin=0.03)
    )


def test_attach_preserves_sources_and_observations():
    a, t, r = fixture()
    original = deepcopy((a, t, r))
    out, tracks, report = run(a, t, r)
    assert report["merged"] == 1
    assert len(out) == len(a)
    assert out[1]["global_track_id"] == "a"
    assert out[1]["pre_stitch_global_track_id"] == "s"
    assert tracks[0]["end_s"] == 13
    assert (a, t, r) == original


def test_competing_anchors_defer():
    a, t, r = fixture()
    r["edges"].append(
        {"left": "b", "right": "s", "score": 0.74, "geometry_quality": 0.8}
    )
    assert run(a, t, r)[2]["merged"] == 0


def test_hard_conflict_cannot_be_overridden():
    a, t, r = fixture()
    r["constraints"] = [
        {"left": "a", "right": "s", "reason": "camera_or_exit_conflict"}
    ]
    assert run(a, t, r)[2]["merged"] == 0


def test_no_merge_option_beats_weak_support():
    a, t, r = fixture()
    r["edges"][0]["score"] = 0.60
    assert run(a, t, r)[2]["merged"] == 0


def test_insertion_requires_both_boundaries():
    a, t, r = fixture()
    t[0]["fragment_ids"].append("b")
    t[0]["last_seen_s"] = 24
    t = t[:2]
    assert run(a, t, r)[2]["merged"] == 0
    r["edges"].append(
        {"left": "s", "right": "b", "score": 0.76, "geometry_quality": 0.8}
    )
    assert run(a, t, r)[2]["merged"] == 1


def test_reachability_alone_never_attaches():
    a, t, r = fixture()
    r["edges"][0]["reason"] = "blind_gap_reachable"
    assert run(a, t, r)[2]["merged"] == 0


def test_two_attachments_respect_mutual_constraints():
    a, t, r = fixture()
    n = {"fragment_id": "s2", "start_s": 11, "end_s": 12}
    r["fragments"].append(n)
    t.append(
        dict(t[1], global_track_id="s2", fragment_ids=["s2"], last_seen_s=12, end_s=12)
    )
    a.append({"global_track_id": "s2", "frame_index": 4})
    r["edges"].append(
        {"left": "a", "right": "s2", "score": 0.8, "geometry_quality": 0.8}
    )
    r["constraints"].append(
        {"left": "s", "right": "s2", "reason": "camera_or_exit_conflict"}
    )
    out, _, report = run(a, t, r)
    assert report["merged"] == 1
    assert out[-1]["global_track_id"] == "a"
    assert out[1]["global_track_id"] == "s"
