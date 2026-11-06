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
    parser.add_argument("--timeout-s", type=float, default=10.0)
    parser.add_argument("--similarity-threshold", type=float, default=0.75)
    parser.add_argument("--ambiguity-margin", type=float, default=0.05)
    parser.add_argument("--min-embedding-confidence", type=float, default=0.50)
    parser.add_argument("--gallery-size", type=int, default=5)
    parser.add_argument("--allow-topology-only-baseline", action="store_true")
    parser.add_argument("--no-offline-reconcile", action="store_true")
    parser.add_argument("--tracklet-threshold", type=float, default=None)
    parser.add_argument("--tracklet-min-samples", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = AssociationConfig(
        timeout_s=args.timeout_s,
        similarity_threshold=args.similarity_threshold,
        ambiguity_margin=args.ambiguity_margin,
        min_embedding_confidence=args.min_embedding_confidence,
        gallery_size=args.gallery_size,
        allow_topology_only_baseline=args.allow_topology_only_baseline,
        offline_reconcile=not args.no_offline_reconcile,
        tracklet_threshold=args.tracklet_threshold,
        tracklet_min_samples=args.tracklet_min_samples,
    )
    cameras = load_cameras(args.cameras)
    observations = load_observations(args.observations)
    exits = load_exit_events(args.exits) if args.exits else []
    engine = AssociationEngine(cameras, config)
    assignments, tracks = engine.run(observations, exits)
    write_results(args.output_dir, assignments, tracks)
    (args.output_dir / "reconciliation.json").write_text(
        json.dumps(
            {
                "offline_reconcile": config.offline_reconcile,
                "tracklet_threshold": config.tracklet_threshold
                if config.tracklet_threshold is not None
                else config.similarity_threshold,
                "merges": engine.reconciliation_log,
            },
            indent=2,
        )
        + "\n"
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
