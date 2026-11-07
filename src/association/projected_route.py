"""Walkable distance between projected floor endpoints on an existing graph.

Virtual endpoint connections are transient. Saved gates, nodes and edges are
never moved. Directed edges are respected; distant points are not snapped.
"""

import itertools
from math import hypot

import numpy as np


def endpoint_route(model, start, finish, max_snap_m):
    start = np.asarray(start, dtype=float) * model.meters_per_unit
    finish = np.asarray(finish, dtype=float) * model.meters_per_unit
    arcs = []
    for a, neighbors in model.graph.items():
        p = np.array([model.nodes[a].x_m, model.nodes[a].y_m])
        for b, distance in neighbors:
            q = np.array([model.nodes[b].x_m, model.nodes[b].y_m])
            vector = q - p
            length = float(np.dot(vector, vector))
            if length > 1e-9:
                arcs.append((a, b, p, q, float(distance), length))
    if not arcs:
        return None

    def snap(point):
        candidates = []
        for a, b, p, q, distance, length in arcs:
            t = float(np.clip(np.dot(point - p, q - p) / length, 0, 1))
            location = p + t * (q - p)
            error = float(np.linalg.norm(point - location))
            candidates.append((error, a, b, t, location, distance))
        closest = min(c[0] for c in candidates)
        if closest > max_snap_m:
            return []
        return [c for c in candidates if c[0] <= closest + 1e-6]

    starts, ends = snap(start), snap(finish)
    best = None
    for sa in starts:
        for sb in ends:
            ea, a, b, ta, pa, da = sa
            eb, c, d, tb, pb, db = sb
            possibilities = []
            if (a, b) == (c, d) and tb >= ta:
                possibilities.append(((tb - ta) * da, [], [pa, pb]))
            middle = model._shortest_path(b, c)
            if middle:
                length, ids = middle
                points = (
                    [pa]
                    + [np.array([model.nodes[n].x_m, model.nodes[n].y_m]) for n in ids]
                    + [pb]
                )
                possibilities.append(
                    ((1 - ta) * da + length + tb * db, list(ids), points)
                )
            for length, ids, points in possibilities:
                total = length + ea + eb
                if best is None or total < best["distance_m"]:
                    vectors = [
                        q - p
                        for p, q in itertools.pairwise(points)
                        if hypot(*(q - p)) > 0.05
                    ]
                    best = {
                        "distance_m": total,
                        "graph_distance_m": length,
                        "source_snap_m": ea,
                        "target_snap_m": eb,
                        "node_ids": ids,
                        "points_m": [p.tolist() for p in points],
                        "first_vector": vectors[0].tolist() if vectors else None,
                        "last_vector": vectors[-1].tolist() if vectors else None,
                    }
    return best
