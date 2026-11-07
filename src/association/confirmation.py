"""Keep every observation; distinguish confirmed episodes from unresolved candidates."""

from collections import defaultdict

from .topology import plausible_recovery


def confirmation_time(observations, config, floor=None):
    """Three coherent independent times within one rolling video-time window."""
    batches = defaultdict(list)
    for o in observations:
        if o.detection_confidence >= 0.25:
            batches[round(o.timestamp_s, 6)].append(o)
    chains = []
    best_count = 0
    for time, batch in sorted(batches.items()):
        chains = [
            [
                o
                for o in chain
                if time - o.timestamp_s <= config.confirmation_window_s + 1e-9
            ]
            for chain in chains
        ]
        chains = [chain for chain in chains if chain]
        proposals = list(chains)
        for o in batch:
            proposals.append([o])
            for chain in chains:
                previous = chain[-1]
                if time - previous.timestamp_s < 0.15 - 1e-9:
                    continue
                coherent = False
                if o.camera_id == previous.camera_id:
                    coherent = plausible_recovery(
                        previous.bbox_xyxy, o.bbox_xyxy, time - previous.timestamp_s
                    )
                elif (
                    floor
                    and floor.point(o) is not None
                    and floor.point(previous) is not None
                ):
                    spatial = floor.between_evidence([previous], [o])
                    coherent = (
                        spatial.adjustment is not None and spatial.adjustment >= 0
                    )
                if coherent:
                    proposal = chain + [o]
                    proposals.append(proposal)
                    if (
                        len(proposal) >= config.confirmation_min_observations
                        and time - proposal[0].timestamp_s
                        >= config.confirmation_min_span_s - 1e-9
                    ):
                        return time, len(proposal)
        # Keep alternative camera paths so simultaneous views do not arbitrarily
        # choose the wrong predecessor. A path still has at most one view per time.
        unique = {tuple(o.reference for o in chain): chain for chain in proposals}
        chains = sorted(
            unique.values(),
            key=lambda chain: (
                -len(chain),
                -(chain[-1].timestamp_s - chain[0].timestamp_s),
                tuple(o.reference for o in chain),
            ),
        )[:16]
        best_count = max((len(chain) for chain in chains), default=0)
    return None, best_count


def annotate_confirmation(observations, assignments, tracks, config, floor=None):
    by_ref = {o.reference: o for o in observations}
    grouped = defaultdict(list)
    for a in assignments:
        grouped[a["global_track_id"]].append(
            by_ref[(a["camera_id"], a["frame_index"], a["local_track_id"])]
        )
    metadata = {}
    for track in tracks:
        time, count = confirmation_time(
            grouped[track["global_track_id"]], config, floor
        )
        track.update(
            confirmation_state="confirmed" if time is not None else "candidate",
            confirmed_at_s=time,
            confirmation_reason="coherent_temporal_support"
            if time is not None
            else "insufficient_coherent_support",
            confirmation_support=count,
        )
        metadata[track["global_track_id"]] = track
    for a in assignments:
        track = metadata[a["global_track_id"]]
        a["confirmation_state"] = track["confirmation_state"]
        a["confirmed_at_s"] = track["confirmed_at_s"]
        o = by_ref[(a["camera_id"], a["frame_index"], a["local_track_id"])]
        a["observed_state"] = (
            "confirmed"
            if track["confirmed_at_s"] is not None
            and o.timestamp_s >= track["confirmed_at_s"]
            else "candidate"
        )
    return (
        assignments,
        tracks,
        {
            "enabled": True,
            "confirmed": sum(t["confirmation_state"] == "confirmed" for t in tracks),
            "candidates": sum(t["confirmation_state"] == "candidate" for t in tracks),
            "min_observations": config.confirmation_min_observations,
            "min_span_s": config.confirmation_min_span_s,
            "window_s": config.confirmation_window_s,
            "candidate_observations": sum(
                a["confirmation_state"] == "candidate" for a in assignments
            ),
        },
    )
