"""Attach short episodes to frozen anchors using the audited fragment graph.

No new identities, deleted detections, anchor merges, or recursive attachments.
"""

from collections import defaultdict
from copy import deepcopy


def stitch_short_tracks(assignments, tracks, report, config):
    report = report or {}
    nodes = {n["fragment_id"]: n for n in report.get("fragments", [])}
    edges = {frozenset((e["left"], e["right"])): e for e in report.get("edges", [])}
    forbidden = {
        frozenset((c["left"], c["right"])) for c in report.get("constraints", [])
    }
    groups = {t["global_track_id"]: set(t.get("fragment_ids", [])) for t in tracks}

    def coverage(ids):
        end, total = float("-inf"), 0
        for start, stop in sorted(
            (nodes[i]["start_s"], nodes[i]["end_s"]) for i in ids
        ):
            total += max(0, stop - max(start, end))
            end = max(end, stop)
        return total

    anchors = [
        t
        for t in tracks
        if groups[t["global_track_id"]] and coverage(groups[t["global_track_id"]]) >= 8
    ]
    shorts = [
        t
        for t in tracks
        if t["last_seen_s"] - t["start_s"] <= 3 and groups[t["global_track_id"]]
    ]
    threshold = max(
        0.58,
        (
            config.tracklet_threshold
            if config.tracklet_threshold is not None
            else config.similarity_threshold
        )
        - 0.03,
    )
    margin = max(0.025, config.ambiguity_margin)

    def evidence(left, right):
        if any(frozenset((a, b)) in forbidden for a in left for b in right):
            return None, "graph_conflict"
        support = []
        for a in left:
            relevant = []
            for b in right:
                e = edges.get(frozenset((a, b)))
                if e is None or e.get("hard_reject"):
                    continue
                # Reachability or unresolved appearance is not identity evidence.
                if e.get("reason") in {
                    "blind_gap_reachable",
                    "unresolved_geometry",
                    "weak_appearance_for_uncertain_geometry",
                }:
                    continue
                if (
                    e.get(
                        "geometry_quality",
                        (e.get("boundary_geometry") or {}).get("reliability", 0),
                    )
                    < 0.15
                ):
                    continue
                relevant.append(e)
            if not relevant:
                return None, "no_direct_geometry"
            # Require support on both sides of an insertion and on every
            # simultaneous view, rather than selecting only one friendly edge.
            n = nodes[a]
            before, after, simultaneous = [], [], []
            for e in relevant:
                other = nodes[e["right"] if e["left"] == a else e["left"]]
                bucket = (
                    before
                    if other["end_s"] < n["start_s"]
                    else after
                    if other["start_s"] > n["end_s"]
                    else simultaneous
                )
                bucket.append(e)
            chosen = [
                max(bucket, key=lambda e: e["score"])
                for bucket in (before, after)
                if bucket
            ]
            chosen += simultaneous
            # If an anchor brackets this fragment, missing evidence on either
            # side must not turn a failed insertion into a one-sided extension.
            if (
                any(nodes[b]["end_s"] < n["start_s"] for b in right) and not before
            ) or (any(nodes[b]["start_s"] > n["end_s"] for b in right) and not after):
                return None, "missing_insertion_boundary"
            if not chosen:
                return None, "no_direct_geometry"
            support.extend(chosen)
        return min(e["score"] for e in support), "direct_boundary_geometry"

    proposals, decisions = [], []
    for short in shorts:
        gid = short["global_track_id"]
        alternatives = []
        for anchor in anchors:
            aid = anchor["global_track_id"]
            score, reason = evidence(groups[gid], groups[aid])
            if score is not None:
                alternatives.append({"target": aid, "score": score, "reason": reason})
        alternatives.sort(key=lambda a: (-a["score"], a["target"]))
        decision = {
            "source": gid,
            "duration_s": short["last_seen_s"] - short["start_s"],
            "alternatives": alternatives,
            "accepted": False,
        }
        decisions.append(decision)
        if not alternatives:
            decision["reason"] = "no_compatible_anchor"
            continue
        best = alternatives[0]
        runner = max(
            threshold, alternatives[1]["score"] if len(alternatives) > 1 else threshold
        )
        if best["score"] - runner < margin:
            decision["reason"] = "ambiguous_or_below_no_merge_margin"
            continue
        proposals.append((best["score"], gid, best["target"], decision))

    # Score-prioritized conflict resolution against already accepted attachments.
    # Every proposal was computed against original anchors, so no cascade occurs.
    accepted, attached = {}, defaultdict(set)
    for score, gid, aid, decision in sorted(
        proposals, key=lambda p: (-p[0], p[1], p[2])
    ):
        incoming = groups[gid]
        if any(frozenset((a, b)) in forbidden for a in incoming for b in attached[aid]):
            decision["reason"] = "attachment_conflict"
            continue
        accepted[gid] = aid
        attached[aid].update(incoming)
        decision.update(
            accepted=True, target=aid, score=score, reason="short_to_frozen_anchor"
        )

    revised = []
    for row in assignments:
        gid = row["global_track_id"]
        revised.append(
            {
                **row,
                **(
                    {
                        "pre_stitch_global_track_id": gid,
                        "global_track_id": accepted[gid],
                        "reason": "short_track_stitch",
                    }
                    if gid in accepted
                    else {}
                ),
            }
        )
    exported = {
        t["global_track_id"]: deepcopy(t)
        for t in tracks
        if t["global_track_id"] not in accepted
    }
    for gid, aid in accepted.items():
        target = exported[aid]
        source = next(t for t in tracks if t["global_track_id"] == gid)
        target["start_s"] = min(target["start_s"], source["start_s"])
        if source["last_seen_s"] > target["last_seen_s"]:
            for key in ("last_seen_s", "end_s", "close_reason", "status"):
                target[key] = source[key]
        target["camera_ids"] = sorted(
            set(target["camera_ids"]) | set(source["camera_ids"])
        )
        target["fragment_ids"] = sorted(
            set(target.get("fragment_ids", [])) | groups[gid]
        )
        target.setdefault("stitched_from", []).append(gid)
    return (
        revised,
        list(exported.values()),
        {
            "enabled": True,
            "short_max_s": 3,
            "anchor_min_observed_s": 8,
            "threshold": threshold,
            "margin": margin,
            "anchors": len(anchors),
            "short_tracks": len(shorts),
            "merged": len(accepted),
            "decisions": decisions,
        },
    )
