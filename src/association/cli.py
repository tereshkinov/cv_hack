from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .engine import AssociationEngine
from .io import load_cameras, load_exit_events, load_observations, write_results
from .models import AssociationConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.association",
        description="Build deterministic global tracks from local camera observations.",
    )
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exits", type=Path)
    parser.add_argument(
        "--project",
        type=Path,
        help="Floor calibration project for soft spatial evidence",
    )
    parser.add_argument("--no-geolocation", action="store_true")
    parser.add_argument(
        "--decision-mode",
        choices=[
            "baseline",
            "balanced",
            "geometry",
            "geometry_visual",
            "geometry_ambiguity",
        ],
        default="baseline",
    )
    parser.add_argument(
        "--fragment-mode", choices=["off", "boundary", "rebuild"], default="off"
    )
    parser.add_argument("--confirm-tracks", action="store_true")
    parser.add_argument("--confirmation-min-observations", type=int, default=3)
    parser.add_argument("--confirmation-min-span-s", type=float, default=0.4)
    parser.add_argument("--confirmation-window-s", type=float, default=1.0)
    parser.add_argument("--boundary-window-s", type=float, default=2.0)
    parser.add_argument("--fragment-beam-width", type=int, default=24)
    parser.add_argument("--geometry-visual-weight", type=float, default=0.08)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--min-embedding-confidence", type=float, default=0.50)
    parser.add_argument("--gallery-size", type=int, default=5)
    parser.add_argument("--allow-topology-only-baseline", action="store_true")
    parser.add_argument("--no-offline-reconcile", action="store_true")
    parser.add_argument("--tracklet-threshold", type=float, default=None)
    parser.add_argument("--adjacency-bonus", type=float, default=0.04)
    parser.add_argument("--tracklet-min-samples", type=int, default=3)
    parser.add_argument("--transition-overlap-tolerance-s", type=float, default=0.75)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AssociationConfig(
        decision_mode=args.decision_mode,
        fragment_mode=args.fragment_mode,
        confirm_tracks=args.confirm_tracks,
        confirmation_min_observations=args.confirmation_min_observations,
        confirmation_min_span_s=args.confirmation_min_span_s,
        confirmation_window_s=args.confirmation_window_s,
        boundary_window_s=args.boundary_window_s,
        fragment_beam_width=args.fragment_beam_width,
        geometry_visual_weight=args.geometry_visual_weight,
        timeout_s=args.timeout_s,
        adjacency_bonus=args.adjacency_bonus,
        similarity_threshold=args.similarity_threshold,
        ambiguity_margin=args.ambiguity_margin,
        min_embedding_confidence=args.min_embedding_confidence,
        gallery_size=args.gallery_size,
        allow_topology_only_baseline=args.allow_topology_only_baseline,
        offline_reconcile=not args.no_offline_reconcile,
        tracklet_threshold=args.tracklet_threshold,
        tracklet_min_samples=args.tracklet_min_samples,
        transition_overlap_tolerance_s=args.transition_overlap_tolerance_s,
    )
    cameras = load_cameras(args.cameras)
    observations = load_observations(args.observations)
    exits = load_exit_events(args.exits) if args.exits else []
    floor_project = (
        json.loads(args.project.read_text())
        if args.project and not args.no_geolocation
        else None
    )
    engine = AssociationEngine(cameras, config, floor_project=floor_project)
    assignments, tracks = engine.run(observations, exits)
    write_results(args.output_dir, assignments, tracks)
    (args.output_dir / "reconciliation.json").write_text(
        json.dumps(
            {
                "decision_mode": config.decision_mode,
                "geometry_visual_weight": config.geometry_visual_weight,
                "confirmation": engine.confirmation_report,
                "fragment_graph": {
                    k: v
                    for k, v in engine.fragment_report.items()
                    if k not in {"fragments", "edges", "constraints"}
                }
                if engine.fragment_report
                else None,
                "gallery_updates_blocked": engine.gallery_updates_blocked,
                "offline_reconcile": config.offline_reconcile,
                "tracklet_threshold": config.tracklet_threshold
                if config.tracklet_threshold is not None
                else config.similarity_threshold,
                "transition_overlap_tolerance_s": config.transition_overlap_tolerance_s,
                "merges": engine.reconciliation_log,
                "merges_scope": "provisional"
                if config.fragment_mode != "off"
                else "final",
                "geolocation": {
                    "enabled": engine.floor is not None,
                    "prediction_model": engine.floor.prediction_model
                    if engine.floor
                    else None,
                    "calibrated_cameras": sorted(engine.floor.calibrations)
                    if engine.floor
                    else [],
                    "calibration_checks": engine.floor.calibration_checks
                    if engine.floor
                    else {},
                    "statistics": dict(engine.floor.stats) if engine.floor else {},
                    "projection_report": engine.floor.projection_report(observations)
                    if engine.floor
                    else None,
                },
                "motion_model": {
                    "enabled": engine.route_model is not None,
                    "scale_source": engine.route_model.scale_source
                    if engine.route_model
                    else None,
                    "meters_per_unit": engine.route_model.meters_per_unit
                    if engine.route_model
                    else None,
                },
            },
            indent=2,
        )
        + "\n"
    )
    if engine.fragment_report is not None:
        (args.output_dir / "fragment_graph.json").write_text(
            json.dumps(engine.fragment_report, ensure_ascii=False, indent=2)
        )
    if args.decision_mode == "balanced":
        selected = {
            (a["camera_id"], a["frame_index"], a["local_track_id"]): a[
                "global_track_id"
            ]
            for a in engine._assignments
        }
        for row in engine.decision_log:
            if row["phase"] == "online":
                # First-pass decision, before offline remapping of provisional IDs.
                row["selected_online"] = (
                    selected.get(
                        (row["camera_id"], row["frame_index"], row["local_track_id"])
                    )
                    == row["candidate_id"]
                )
        (args.output_dir / "decisions.jsonl").write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in engine.decision_log
            )
        )
    print(
        json.dumps(
            {
                "assignments": len(assignments),
                "global_tracks": len(tracks),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
