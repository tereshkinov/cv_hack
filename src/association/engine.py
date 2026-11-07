from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from math import sqrt
from statistics import fmean

from .models import AssociationConfig, Camera, ExitEvent, Observation, TrackState
from .topology import CameraGraph, plausible_recovery


class AssociationEngine:
    """Deterministic offline association with a global track lifecycle."""

    def __init__(
        self,
        cameras: Iterable[Camera],
        config: AssociationConfig | None = None,
        floor_project: dict | None = None,
    ) -> None:
        camera_list = list(cameras)
        self.cameras = {camera.camera_id: camera for camera in camera_list}
        if len(self.cameras) != len(camera_list):
            raise ValueError("camera_id values must be unique")
        if not self.cameras:
            raise ValueError("at least one camera is required")
        self._validate_camera_links()
        self.config = config or AssociationConfig()
        self.floor = None
        self.route_model = None
        self.boundary_geometry = None
        self._floor_history = defaultdict(list)
        if floor_project:
            from .boundary_geometry import BoundaryGeometryMatcher
            from .geolocation import FloorMatcher
            from .spatial_motion import OfficeMotionModel

            self.floor = FloorMatcher(floor_project, camera_list)
            self.route_model = OfficeMotionModel.from_project(floor_project)
            self.boundary_geometry = BoundaryGeometryMatcher(
                camera_list,
                floor_matcher=self.floor,
                route_model=self.route_model,
            )
        self.graph = CameraGraph(self.cameras)
        self._overlaps = self.graph.overlaps
        self._tracks: dict[str, TrackState] = {}
        self._local_to_global: dict[tuple[str, str], str] = {}
        self._assignments: list[dict[str, object]] = []
        self._next_id = 1
        self.reconciliation_log: list[dict] = []

    def run(
        self,
        observations: Iterable[Observation],
        exit_events: Iterable[ExitEvent] = (),
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if self._assignments or self._tracks:
            raise RuntimeError("AssociationEngine instances are single-use")

        observations = list(observations)
        exit_events = list(exit_events)
        grouped_observations: dict[float, list[Observation]] = defaultdict(list)
        grouped_exits: dict[float, list[ExitEvent]] = defaultdict(list)
        references: set[tuple[str, int, str]] = set()

        for observation in observations:
            self._validate_observation(observation)
            if observation.reference in references:
                raise ValueError(
                    f"duplicate observation reference: {observation.reference}"
                )
            references.add(observation.reference)
            grouped_observations[observation.timestamp_s].append(observation)
        for event in exit_events:
            if event.camera_id not in self.cameras:
                raise ValueError(f"unknown camera_id in exit event: {event.camera_id}")
            grouped_exits[event.timestamp_s].append(event)

        self._local_intervals = {}
        for o in observations:
            start, end = self._local_intervals.get(
                o.local_key, (o.timestamp_s, o.timestamp_s)
            )
            self._local_intervals[o.local_key] = (
                min(start, o.timestamp_s),
                max(end, o.timestamp_s),
            )
        timestamps = sorted(set(grouped_observations) | set(grouped_exits))
        for timestamp in timestamps:
            self._advance_time(timestamp)
            self._process_timestamp(grouped_observations.get(timestamp, []))
            for event in sorted(
                grouped_exits.get(timestamp, []),
                key=lambda item: (item.camera_id, item.local_track_id),
            ):
                self._process_exit(event)

        self._finalize_open_tracks()
        assignments = list(self._assignments)
        tracks = [
            self._export_track(self._tracks[track_id])
            for track_id in sorted(self._tracks)
        ]
        if self.config.offline_reconcile:
            from .reconcile import reconcile

            assignments, tracks, self.reconciliation_log = reconcile(
                self, observations, assignments, tracks, exit_events
            )
        return assignments, tracks

    def _validate_camera_links(self) -> None:
        known = set(self.cameras)
        for camera in self.cameras.values():
            unknown = (set(camera.overlap_with) | set(camera.transition_to)) - known
            if unknown:
                raise ValueError(
                    f"camera {camera.camera_id!r} references unknown cameras: {sorted(unknown)}"
                )
            if (
                camera.camera_id in camera.overlap_with
                or camera.camera_id in camera.transition_to
            ):
                raise ValueError(f"camera {camera.camera_id!r} cannot link to itself")

    def _validate_observation(self, observation: Observation) -> None:
        if observation.camera_id not in self.cameras:
            raise ValueError(
                f"unknown camera_id in observation: {observation.camera_id}"
            )

    def _advance_time(self, timestamp: float) -> None:
        for track in self._tracks.values():
            if track.status == "closed":
                continue
            gap = timestamp - track.last_seen_s
            if gap < 0:
                raise ValueError("timestamps must be processable in ascending order")
            if gap > self.config.timeout_s:
                self._close(track, "timeout")
            elif gap > 0:
                track.status = "pending"

    def _process_timestamp(self, observations: list[Observation]) -> None:
        by_camera: dict[str, list[Observation]] = defaultdict(list)
        for observation in observations:
            by_camera[observation.camera_id].append(observation)

        for camera_id in sorted(by_camera):
            camera_observations = sorted(
                by_camera[camera_id],
                key=lambda item: (item.frame_index, item.local_track_id),
            )
            unmatched: list[Observation] = []
            used_tracks: set[str] = set()

            for observation in camera_observations:
                mapped_id = self._local_to_global.get(observation.local_key)
                mapped = self._tracks.get(mapped_id) if mapped_id else None
                if mapped is not None and mapped.status != "closed":
                    self._assign(observation, mapped, None, "local_track_continuity")
                    used_tracks.add(mapped.global_track_id)
                else:
                    unmatched.append(observation)

            candidate_map = {
                observation.reference: self._candidates(observation, used_tracks)
                for observation in unmatched
            }
            edges: list[tuple[float, str, str, Observation, TrackState, str]] = []
            for observation in unmatched:
                candidates = candidate_map[observation.reference]
                if self._is_ambiguous(candidates):
                    continue
                for score, track, reason in candidates:
                    edges.append(
                        (
                            -score,
                            observation.local_track_id,
                            track.global_track_id,
                            observation,
                            track,
                            reason,
                        )
                    )

            matched_observations: set[tuple[str, int, str]] = set()
            for negative_score, _, _, observation, track, reason in sorted(edges):
                if observation.reference in matched_observations:
                    continue
                if track.global_track_id in used_tracks:
                    continue
                self._assign(observation, track, -negative_score, reason)
                matched_observations.add(observation.reference)
                used_tracks.add(track.global_track_id)

            for observation in unmatched:
                if observation.reference not in matched_observations:
                    self._assign(
                        observation, self._new_track(observation), None, "new_track"
                    )

    def _candidates(
        self, observation: Observation, used_tracks: set[str]
    ) -> list[tuple[float, TrackState, str]]:
        candidates: list[tuple[float, TrackState, str]] = []
        for track in self._tracks.values():
            if track.status == "closed" or track.global_track_id in used_tracks:
                continue
            topology_reason = self._topology_reason(observation, track)
            if topology_reason is None:
                continue
            score = self._appearance_score(observation, track)
            if score is not None:
                score = min(
                    1.0,
                    score
                    + self.graph.bonus(topology_reason, self.config.adjacency_bonus),
                )
            geo = (
                self.floor.online(
                    observation, self._floor_history[track.global_track_id]
                )
                if self.floor
                else None
            )
            if geo is not None and score is not None:
                self.floor.stats["online_evaluations"] += 1
                # Appearance still needs meaningful support even with a perfect location.
                score = min(1.0, score + geo) if score >= 0.45 else score
            if score is not None and score >= self.config.similarity_threshold:
                candidates.append(
                    (
                        score,
                        track,
                        f"{topology_reason}_reid"
                        + ("_floor" if geo is not None else ""),
                    )
                )
            elif (
                score is None
                and observation.embedding is None
                and topology_reason != "same_camera"
                and self.config.allow_topology_only_baseline
            ):
                candidates.append((0.0, track, f"{topology_reason}_topology_baseline"))
        return sorted(candidates, key=lambda item: (-item[0], item[1].global_track_id))

    def _is_ambiguous(self, candidates: list[tuple[float, TrackState, str]]) -> bool:
        if len(candidates) < 2:
            return False
        best, second = candidates[0][0], candidates[1][0]
        return best - second < self.config.ambiguity_margin

    def _appearance_score(
        self, observation: Observation, track: TrackState
    ) -> float | None:
        if observation.embedding is None:
            return None
        if observation.detection_confidence < self.config.min_embedding_confidence:
            return None
        if track.embedding_model_id != observation.embedding_model_id:
            return None
        compatible = [
            vector
            for _, vector in track.embedding_gallery
            if len(vector) == len(observation.embedding)
        ]
        if not compatible:
            return None
        similarities = sorted(
            (
                _cosine_similarity(observation.embedding, vector)
                for vector in compatible
            ),
            reverse=True,
        )
        return fmean(similarities[:3])

    def _topology_reason(
        self, observation: Observation, track: TrackState
    ) -> str | None:
        target = observation.camera_id
        # A new local fragment may recover this camera's ID only after a real gap.
        if target in track.camera_last_seen:
            gap = observation.timestamp_s - track.camera_last_seen[target]
            if gap <= 0:
                return None
            start, end = self._local_intervals[observation.local_key]
            for key, gid in self._local_to_global.items():
                if gid != track.global_track_id or key[0] != target:
                    continue
                old_start, old_end = self._local_intervals[key]
                if max(start, old_start) <= min(end, old_end):
                    return None
            if gap <= self.config.timeout_s:
                if plausible_recovery(
                    track.camera_last_bbox[target], observation.bbox_xyxy, gap
                ):
                    return "same_camera"
                return None
            # A long absence on this camera is valid if another recent camera
            # continuously carries the episode and has a permitted edge back.
        # Use recent graph edges, not a camera the person visited long ago.
        reasons = []
        for source, last_seen in sorted(track.camera_last_seen.items()):
            gap = observation.timestamp_s - last_seen
            if not 0 <= gap <= self.config.timeout_s:
                continue
            relation = self.graph.relation(source, target, gap)
            if relation:
                reasons.append(relation)
        if "overlap" in reasons:
            return "overlap"
        return "transition" if "transition" in reasons else None

    def _new_track(self, observation: Observation) -> TrackState:
        global_track_id = f"g{self._next_id}"
        self._next_id += 1
        track = TrackState(
            global_track_id=global_track_id,
            start_s=observation.timestamp_s,
            last_seen_s=observation.timestamp_s,
        )
        self._tracks[global_track_id] = track
        return track

    def _assign(
        self,
        observation: Observation,
        track: TrackState,
        score: float | None,
        reason: str,
    ) -> None:
        track.status = "active"
        track.last_seen_s = max(track.last_seen_s, observation.timestamp_s)
        track.camera_ids.add(observation.camera_id)
        track.camera_last_seen[observation.camera_id] = max(
            track.camera_last_seen.get(observation.camera_id, observation.timestamp_s),
            observation.timestamp_s,
        )
        track.camera_last_bbox[observation.camera_id] = observation.bbox_xyxy
        self._local_to_global[observation.local_key] = track.global_track_id
        self._update_gallery(track, observation)
        if self.floor:
            point = self.floor.point(observation)
            history = self._floor_history[track.global_track_id]
            if point is not None:
                # Keep distinct times; do not derive velocity from two camera views at one time.
                sample = (observation.timestamp_s, *point)
                if history and abs(history[-1][0] - sample[0]) < 1e-6:
                    history[-1] = sample
                elif not history or sample[0] - history[-1][0] >= 0.15:
                    history.append(sample)
                history[:] = [p for p in history if sample[0] - p[0] <= 2]
        assignment: dict[str, object] = {
            "schema_version": 1,
            "camera_id": observation.camera_id,
            "frame_index": observation.frame_index,
            "local_track_id": observation.local_track_id,
            "global_track_id": track.global_track_id,
            "reason": reason,
        }
        if score is not None:
            assignment["match_score"] = round(score, 8)
        self._assignments.append(assignment)

    def _update_gallery(self, track: TrackState, observation: Observation) -> None:
        if observation.embedding is None:
            return
        if observation.detection_confidence < self.config.min_embedding_confidence:
            return
        if track.embedding_model_id is None:
            track.embedding_model_id = observation.embedding_model_id
        if track.embedding_model_id != observation.embedding_model_id:
            return
        if track.embedding_gallery and len(track.embedding_gallery[0][1]) != len(
            observation.embedding
        ):
            return
        track.embedding_gallery.append(
            (observation.detection_confidence, observation.embedding)
        )
        track.embedding_gallery.sort(key=lambda item: item[0], reverse=True)
        del track.embedding_gallery[self.config.gallery_size :]

    def _process_exit(self, event: ExitEvent) -> None:
        global_id = self._local_to_global.get((event.camera_id, event.local_track_id))
        if global_id is None:
            raise ValueError(
                "explicit exit references an unseen local track: "
                f"{event.camera_id}/{event.local_track_id}"
            )
        track = self._tracks[global_id]
        if track.status != "closed":
            self._close(track, "explicit_exit")

    @staticmethod
    def _close(track: TrackState, reason: str) -> None:
        track.status = "closed"
        track.end_s = track.last_seen_s
        track.close_reason = reason

    def _finalize_open_tracks(self) -> None:
        for track in self._tracks.values():
            if track.status != "closed":
                self._close(track, "end_of_video")

    @staticmethod
    def _export_track(track: TrackState) -> dict[str, object]:
        return {
            "global_track_id": track.global_track_id,
            "start_s": track.start_s,
            "last_seen_s": track.last_seen_s,
            "end_s": track.end_s,
            "status": track.status,
            "camera_ids": sorted(track.camera_ids),
            "close_reason": track.close_reason,
        }


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))
