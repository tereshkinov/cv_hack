"""Delayed cross-view trajectory evidence and exact small handoff assignments.

The result is an offline fragment decision: the first observation is relabeled
retroactively. No claim of streaming latency or globally optimal identity graph.
"""

import itertools
from collections import defaultdict
from math import exp, hypot, sqrt
from statistics import median

import numpy as np

WINDOW_S = 1.5


def _samples(floor, rows):
    result = []
    for o in sorted(rows, key=lambda o: o.timestamp_s):
        p = floor.projection(o)
        if not p.available or p.reliability < 0.3:
            continue
        if result and o.timestamp_s - result[-1][0] < 0.12:
            continue
        result.append((o.timestamp_s, *p.point, p.reliability))
    return result


def _fit(samples):
    if len(samples) < 3 or samples[-1][0] - samples[0][0] < 0.4 - 1e-8:
        return None
    vx = []
    vy = []
    for i, a in enumerate(samples):
        for b in samples[i + 1 :]:
            dt = b[0] - a[0]
            if dt >= 0.25:
                vx.append((b[1] - a[1]) / dt)
                vy.append((b[2] - a[2]) / dt)
    if not vx:
        return None
    velocity = np.array([median(vx), median(vy)])
    t0 = samples[0][0]
    origin = np.median(
        [np.array(p[1:3]) - velocity * (p[0] - t0) for p in samples], axis=0
    )
    residual = median(
        hypot(*(np.array(p[1:3]) - origin - velocity * (p[0] - t0))) for p in samples
    )
    return t0, origin, velocity, residual


def _position(fit, time):
    t, p, v, _ = fit
    return p + v * (time - t)


def _cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def trajectory_evidence(engine, a, b, boundary=None):
    from .handoff_portals import portal_evidence

    portal, portal_details = portal_evidence(engine, a, b)
    if portal is not None:
        return portal, portal_details
    floor = engine.floor
    if floor is None or a.camera_id == b.camera_id:
        return None, {"reason": "handoff_not_applicable"}
    overlap = min(a.end_s, b.end_s) - max(a.start_s, b.start_s)
    simultaneous = overlap >= 0
    # A touch or brief overlap is a handoff boundary, not a shared-view window.
    boundary_touch = simultaneous and overlap < 0.4 - 1e-8
    gap = max(0, b.start_s - a.end_s)
    if gap > engine.config.timeout_s:
        return None, {"reason": "handoff_outside_timeout"}
    if simultaneous and not boundary_touch:
        begin = max(a.start_s, b.start_s)
        end = min(a.end_s, b.end_s, begin + WINDOW_S)
        left = _samples(
            floor, [o for o in a.observations if begin <= o.timestamp_s <= end]
        )
        right = _samples(
            floor, [o for o in b.observations if begin <= o.timestamp_s <= end]
        )
    else:
        left = _samples(floor, a.tail(WINDOW_S))
        right = _samples(floor, b.head(WINDOW_S))
    fa, fb = _fit(left), _fit(right)
    if fa is None or fb is None:
        return None, {
            "reason": "handoff_pending_trajectory",
            "left_samples": len(left),
            "right_samples": len(right),
            "lookahead_window_s": WINDOW_S,
        }
    metric = floor.metric_scale
    mpu = metric["meters_per_unit"] if metric else None
    # Cross-view localization error is wider than a single track's fit jitter.
    radius = max(floor.radius, 0.55 / mpu if mpu else floor.radius)
    calibration_uncertainty = (
        max(0.15, metric["uncertainty_fraction"]) if metric else 0.25
    )
    moving_a = np.linalg.norm(fa[2]) * (left[-1][0] - left[0][0]) > max(
        radius * 0.5, fa[3] * 3
    )
    moving_b = np.linalg.norm(fb[2]) * (right[-1][0] - right[0][0]) > max(
        radius * 0.5, fb[3] * 3
    )
    cosine = _cosine(fa[2], fb[2]) if moving_a and moving_b else None
    route = (
        boundary
        if boundary and boundary.available and boundary.reason.startswith("route:")
        else None
    )
    details = {
        "reason": "handoff_trajectory_boundary"
        if boundary_touch
        else "handoff_trajectory_overlap"
        if simultaneous
        else "handoff_trajectory_transition",
        "left_samples": len(left),
        "right_samples": len(right),
        "lookahead_window_s": WINDOW_S,
        "appearance_used": False,
        "direction_cosine": cosine,
        "left_velocity": fa[2].tolist(),
        "right_velocity": fb[2].tolist(),
        "metric_scale": metric,
    }
    if boundary_touch:
        # Both local paths need temporal support, but at different sides of the
        # handoff. Never extrapolate past their observed boundary endpoints.
        lo, hi = b.start_s, a.end_s
        times = np.linspace(lo, hi, 3) if hi - lo > 1e-8 else [lo]
        distances = [
            float(np.linalg.norm(_position(fa, t) - _position(fb, t))) for t in times
        ]
        speed = max(np.linalg.norm(fa[2]), np.linalg.norm(fb[2]))
        # Both cameras contribute localization uncertainty. The boundary has
        # no shared interval that could average down either projection error.
        position_tolerance = sqrt(2) * radius
        tolerance = (
            position_tolerance + floor.sync_uncertainty_s * speed + fa[3] + fb[3]
        )
        discrepancy = (median(distances) * 0.7 + max(distances) * 0.3) / tolerance
        if cosine is not None:
            discrepancy = sqrt(discrepancy**2 + 0.5 * max(0, -cosine) ** 2)
        details.update(
            common_span_s=max(0, overlap),
            position_residual=median(distances),
            comparison_radius=tolerance,
            combined_position_tolerance=position_tolerance,
            evidence_window="tail_to_head",
            left_span_s=left[-1][0] - left[0][0],
            right_span_s=right[-1][0] - right[0][0],
        )
    elif simultaneous:
        # Fit each view independently, then compare at the same video times.
        lo = max(left[0][0], right[0][0])
        hi = min(left[-1][0], right[-1][0])
        if hi - lo < 0.4 - 1e-8:
            return None, details | {"reason": "handoff_pending_trajectory"}
        times = np.linspace(lo, hi, 5)
        distances = [
            float(np.linalg.norm(_position(fa, t) - _position(fb, t))) for t in times
        ]
        speed = max(np.linalg.norm(fa[2]), np.linalg.norm(fb[2]))
        tolerance = radius + floor.sync_uncertainty_s * speed + fa[3] + fb[3]
        discrepancy = (median(distances) * 0.7 + max(distances) * 0.3) / tolerance
        if cosine is not None:
            discrepancy = sqrt(discrepancy**2 + 0.8 * max(0, -cosine) ** 2)
        details.update(
            common_span_s=hi - lo,
            position_residual=median(distances),
            comparison_radius=tolerance,
        )
    elif route is not None:
        # Estimate travel from actual observed floor endpoints on the existing
        # walkable graph. Camera gate positions are only a fallback.
        dynamic_route = None
        if engine.route_model and metric:
            from .projected_route import endpoint_route

            dynamic_route = endpoint_route(
                engine.route_model,
                _position(fa, a.end_s),
                _position(fb, b.start_s),
                max(0.8, 2 * radius * mpu),
            )
        if dynamic_route is not None:
            model = engine.route_model
            distance = dynamic_route["distance_m"]
            speed = (
                float(np.linalg.norm(fa[2])) * mpu if moving_a else model.typical_speed
            )
            speed = max(model.min_speed, min(model.max_speed, speed))
            expected = distance / speed
            sigma = max(
                model.timing_sigma_s,
                2 * radius * mpu / speed,
                expected * max(0.25, calibration_uncertainty),
            )
            # Slow arrivals can include a stop. Fast arrivals remain penalized,
            # with tolerance for uncertain metric scale and endpoint projection.
            residual = max(expected - gap, 0.2 * (gap - expected))
            discrepancy = residual / sigma
            alignments = []
            if moving_a and dynamic_route["first_vector"]:
                alignments.append(_cosine(fa[2], dynamic_route["first_vector"]))
            if moving_b and dynamic_route["last_vector"]:
                alignments.append(_cosine(fb[2], dynamic_route["last_vector"]))
            if alignments:
                discrepancy = sqrt(discrepancy**2 + 0.6 * max(0, -min(alignments)) ** 2)
            details.update(
                projected_route=dynamic_route,
                expected_travel_s=expected,
                travel_sigma_s=sigma,
                route_tangent_cosines=alignments,
                route_reason="observed_floor_endpoints",
            )
        else:
            discrepancy, route_details = _gate_route_discrepancy(
                route, fa, fb, moving_a, moving_b
            )
            details.update(route_details)
    elif gap <= 4:
        mid = (a.end_s + b.start_s) / 2
        delta = float(np.linalg.norm(_position(fa, mid) - _position(fb, mid)))
        tolerance = (
            radius
            + fa[3]
            + fb[3]
            + floor.speed * (0.20 * gap + floor.sync_uncertainty_s)
        )
        discrepancy = delta / tolerance
        if cosine is not None:
            discrepancy = sqrt(discrepancy**2 + 0.5 * max(0, -cosine) ** 2)
        details.update(position_residual=delta, comparison_radius=tolerance)
    else:
        return None, details | {"reason": "handoff_long_gap_without_route"}
    quality = min(median(p[3] for p in left), median(p[3] for p in right))
    quality *= 0.8 if not floor.sync_verified else 1
    quality *= max(0.5, 1 - calibration_uncertainty)
    if boundary_touch:
        quality *= 0.85
    if not simultaneous and route:
        quality = min(quality, route.reliability)
    details.update(discrepancy=float(discrepancy), geometry_quality=float(quality))
    return (
        exp(-0.5 * discrepancy**2),
        float(quality),
        float(discrepancy),
        details["reason"],
        min(len(left), len(right)),
    ), details


def _gate_route_discrepancy(route, fa, fb, moving_a, moving_b):
    trajectory = (route.expected_region or {}).get("trajectory", [])
    vectors = []
    for p, q in itertools.pairwise(trajectory):
        v = np.array([q["x_m"] - p["x_m"], q["y_m"] - p["y_m"]])
        if np.linalg.norm(v) > 0.05:
            vectors.append(v)
    alignments = []
    if vectors and moving_a:
        alignments.append(_cosine(fa[2], vectors[0]))
    if vectors and moving_b:
        alignments.append(_cosine(fb[2], vectors[-1]))
    discrepancy = abs(route.discrepancy or 0)
    if alignments:
        discrepancy = sqrt(discrepancy**2 + 0.6 * max(0, -min(alignments)) ** 2)
    return discrepancy, {
        "route_tangent_cosines": alignments,
        "route_reason": route.reason,
    }


def solve_matching(sources, targets, weights, ban=None):
    """Exact assignment with a zero-valued 'no match' option for each target."""
    states = {0: (0, ())}
    for target in targets:
        next_states = {}
        for mask, (score, pairs) in states.items():
            choices = [(mask, score, pairs)]
            for i, source in enumerate(sources):
                key = (source, target)
                if mask & (1 << i) or key == ban or weights.get(key, 0) <= 0:
                    continue
                choices.append((mask | (1 << i), score + weights[key], pairs + (key,)))
            for new_mask, value, links in choices:
                old = next_states.get(new_mask)
                if (
                    old is None
                    or value > old[0] + 1e-12
                    or (abs(value - old[0]) <= 1e-12 and links < old[1])
                ):
                    next_states[new_mask] = (value, links)
        states = next_states
    return min(states.values(), key=lambda p: (-p[0], p[1]))


def joint_handoffs(nodes, cache, threshold, margin):
    """Certify small simultaneous competitor packets; leave other graph edges alone."""
    by_id = {n.fragment_id: n for n in nodes}
    groups = defaultdict(list)
    for (aid, bid), (score, details) in cache.items():
        a, b = by_id[aid], by_id[bid]
        if (
            a.camera_id != b.camera_id
            and score is not None
            and not details.get("hard_reject")
            and details.get("handoff", {})
            .get("reason", "")
            .startswith("handoff_trajectory")
        ):
            groups[a.camera_id, b.camera_id].append((aid, bid))
    decisions = {}
    packets = []
    for cameras, edges in sorted(groups.items()):
        packets_edges = []
        for edge in sorted(edges, key=lambda p: (by_id[p[1]].start_s, p)):
            if (
                not packets_edges
                or by_id[edge[1]].start_s - by_id[packets_edges[-1][0][1]].start_s > 2
            ):
                packets_edges.append([])
            packets_edges[-1].append(edge)
        for edges in packets_edges:
            sources = sorted({a for a, b in edges})
            targets = sorted({b for a, b in edges})
            if not 2 <= len(sources) <= 10 or not 2 <= len(targets) <= 10:
                continue
            if any(
                min(by_id[i].end_s for i in ids) - max(by_id[i].start_s for i in ids)
                < 0.2
                for ids in [sources, targets]
            ):
                continue
            weights = {edge: cache[edge][0] - threshold for edge in edges}
            best, links = solve_matching(sources, targets, weights)
            stable = {}
            for link in links:
                alternative, _ = solve_matching(sources, targets, weights, ban=link)
                stable[link] = best - alternative
            accepted = {
                link for link, diff in stable.items() if diff >= max(0.03, margin)
            }
            for edge in edges:
                decisions[edge] = {
                    "selected": edge in accepted,
                    "assignment_margin": stable.get(edge),
                    "packet": len(packets),
                }
            packets.append(
                {
                    "cameras": list(cameras),
                    "sources": sources,
                    "targets": targets,
                    "objective": best,
                    "solver": "exact_bitmask_assignment",
                    "decisions": [
                        {"left": a, "right": b, **decisions[a, b]} for a, b in edges
                    ],
                }
            )
    return decisions, packets
