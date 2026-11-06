"""Decode videos, run an isolated tracker per camera, export observations."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
import heapq
from contextlib import ExitStack
import math
from pathlib import Path
import platform
import resource
import time
from importlib.metadata import version

import cv2
import numpy as np

from .embeddings import ColorHistogram, OSNet, normalized, sha256


@dataclass
class Options:
    detector: str = 'models/yolo11n.pt'
    device: str = 'cpu'
    embedding: str = 'osnet'
    reid_weights: str = 'models/osnet_x0_25_msmt17.pth'
    image_size: int = 640
    confidence: float = 0.1
    frame_stride: int = 1
    local_buffer_s: float = 1.0
    embedding_interval_s: float = 0.5
    min_crop_height: int = 48
    min_sharpness: float = 10.0
    max_overlap: float = 0.6
    top_k: int = 5
    max_frames: int | None = None
    preview: bool = False


def load_cameras(path: Path, root: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if raw.get('schema_version') != 1:
        raise ValueError('Camera config must have schema_version=1')
    cameras = raw.get('cameras', [])
    if not cameras:
        raise ValueError('No cameras in configuration')
    seen = set()
    for camera in cameras:
        cid = camera.get('camera_id')
        if not isinstance(cid, str) or not cid or cid in seen:
            raise ValueError('camera_id must be a unique nonempty string')
        seen.add(cid)
        offset = camera.get('time_offset_s', 0)
        if not isinstance(offset, (int, float)) or not math.isfinite(offset):
            raise ValueError(f'Invalid time_offset_s for {cid}')
        video = Path(camera['video_path'])
        if not video.is_absolute():
            video = root / video
        if not video.is_file():
            raise FileNotFoundError(f'{cid}: video not found: {video}')
    return cameras


def clip_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    box = [max(0.0, min(width, x1)), max(0.0, min(height, y1)),
           max(0.0, min(width, x2)), max(0.0, min(height, y2))]
    return box if box[2] > box[0] and box[3] > box[1] else None


def crop_quality(frame, box, other_boxes, options):
    x1, y1, x2, y2 = box
    ix1, iy1, ix2, iy2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.shape[0] < options.min_crop_height or crop.shape[1] < 8:
        return None
    area = (x2 - x1) * (y2 - y1)
    # Bounding-box overlap is a conservative proxy, not a segmentation/occlusion estimate.
    overlap = max((max(0, min(x2, b[2]) - max(x1, b[0])) *
                   max(0, min(y2, b[3]) - max(y1, b[1])) / area for b in other_boxes), default=0)
    sharpness = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if overlap > options.max_overlap or sharpness < options.min_sharpness:
        return None
    quality = math.sqrt(area) * math.log1p(sharpness) * (1 - overlap)
    return crop.copy(), quality


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_options(o):
    if o.embedding not in {'osnet', 'color', 'none'}:
        raise ValueError('embedding must be osnet, color, or none')
    if o.frame_stride < 1 or o.top_k < 1 or o.image_size < 32:
        raise ValueError('frame_stride/top_k/image_size must be positive')
    if not 0 < o.confidence <= 1 or not 0 <= o.max_overlap <= 1:
        raise ValueError('Invalid confidence or max_overlap')
    for x in (o.local_buffer_s, o.embedding_interval_s, o.min_sharpness):
        if not math.isfinite(x) or x < 0:
            raise ValueError('Timing and sharpness settings must be finite and nonnegative')
    if o.min_crop_height < 1 or (o.max_frames is not None and o.max_frames < 1):
        raise ValueError('Crop height/max_frames must be positive')


def run(cameras_path: Path, output: Path, root: Path, options: Options,
        detector_factory=None, encoder=None):
    validate_options(options)
    cameras = load_cameras(cameras_path, root)
    # Prevent accidental overwrite of a previous or partial processing run.
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        if detector_factory is None:
            from ultralytics import YOLO
            detector_path = Path(options.detector)
            if not detector_path.is_absolute():
                detector_path = root / detector_path
            if not detector_path.is_file():
                raise FileNotFoundError(f'Detector missing: {detector_path}. Run python -m src.vision prepare')
            detector_factory = lambda: YOLO(str(detector_path))
        if encoder is None:
            if options.embedding == 'osnet':
                weights = Path(options.reid_weights)
                encoder = OSNet(weights if weights.is_absolute() else root / weights, options.device)
            elif options.embedding == 'color':
                encoder = ColorHistogram()
        summaries, tracklets, observation_count = [], [], 0
        camera_files = []
        for camera_number, camera in enumerate(cameras):
            camera_file = output / f'observations-camera-{camera_number}.jsonl'
            camera_files.append(camera_file)
            with camera_file.open('w') as stream:
                summary, local, count = process_camera(camera, root, output, options,
                                                       detector_factory(), encoder, stream)
            summary['observations_file'] = camera_file.name
            summaries.append(summary)
            tracklets.extend(local)
            observation_count += count
            print(f"{camera['camera_id']}: {count} observations, {len(local)} local tracks", flush=True)
        # Per-camera sequences are ordered; merge with bounded memory for temporal consumers.
        with ExitStack() as stack, (output / 'observations.jsonl').open('w') as merged:
            streams = [map(json.loads, stack.enter_context(p.open())) for p in camera_files]
            for record in heapq.merge(*streams, key=lambda o: (o['timestamp_s'], o['camera_id'], o['local_track_id'])):
                merged.write(json.dumps(record, allow_nan=False) + '\n')
        write_json(output / 'tracklets.json', {'schema_version': 1, 'scope': 'local_camera_only', 'tracklets': tracklets})
        elapsed = time.perf_counter() - started
        report = {'schema_version': 1, 'status': 'complete', 'options': asdict(options),
                  'cameras': summaries, 'observation_count': observation_count,
                  'wall_time_s': elapsed,
                  'processed_fps': sum(c['processed_frames'] for c in summaries) / elapsed,
                  'peak_process_rss_mb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if platform.system() == 'Darwin' else 1024),
                  'platform': platform.platform(), 'embedding_model_id': getattr(encoder, 'model_id', None),
                  'embedding_dimension': getattr(encoder, 'dimension', None),
                  'embedding_normalization': 'L2' if encoder else None,
                  'packages': {p: version(p) for p in ['ultralytics', 'torch', 'opencv-python', 'numpy']}}
        weights_path = Path(options.detector)
        if not weights_path.is_absolute():
            weights_path = root / weights_path
        if weights_path.is_file():
            report['detector_sha256'] = sha256(weights_path)
        write_json(output / 'run.json', report)
        return report
    except Exception as error:
        write_json(output / 'FAILED.json', {'status': 'failed', 'error': str(error)})
        raise


def process_camera(camera, root, output, options, detector, encoder, stream):
    video = Path(camera['video_path'])
    if not video.is_absolute():
        video = root / video
    cap = cv2.VideoCapture(str(video))
    writer = None
    local = {}
    count = processed = decoded = 0
    start = time.perf_counter()
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f'Invalid video metadata: {video}')
        # Pinned Ultralytics ByteTrack uses a buffer measured in tracker update steps.
        tracker_config = output / f'tracker-{len(list(output.glob("tracker-*.yaml")))}.yaml'
        tracker_config.write_text('tracker_type: bytetrack\ntrack_high_thresh: 0.25\n'
            'track_low_thresh: 0.1\nnew_track_thresh: 0.25\n'
            f'track_buffer: {max(1, math.ceil(options.local_buffer_s * fps / options.frame_stride))}\n'
            'match_thresh: 0.8\nfuse_score: true\n')
        if options.preview:
            preview_path = output / f'preview-{len(list(output.glob("preview-*.mp4")))}.mp4'
            writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                     fps / options.frame_stride, (width, height))
            if not writer.isOpened():
                raise ValueError(f'Cannot create preview: {preview_path}')
        while options.max_frames is None or decoded < options.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index = decoded
            decoded += 1
            if frame_index % options.frame_stride:
                continue
            processed += 1
            timestamp = frame_index / fps + camera.get('time_offset_s', 0)
            if timestamp < 0:
                continue
            results = detector.track(frame, persist=True, tracker=str(tracker_config),
                                     classes=[0], conf=options.confidence, imgsz=options.image_size,
                                     device=options.device, verbose=False)
            boxes = results[0].boxes
            records = []
            if boxes is not None and boxes.id is not None:
                for coords, identity, confidence in zip(boxes.xyxy.cpu().numpy(),
                        boxes.id.cpu().numpy(), boxes.conf.cpu().numpy()):
                    box = clip_box(coords, width, height)
                    if box is not None:
                        records.append((f'l{int(identity)}', box, float(confidence)))
            records.sort(key=lambda record: record[0])
            pending = []
            for identity, box, confidence in records:
                state = local.setdefault(identity, {'start_s': timestamp, 'last_seen_s': timestamp,
                            'count': 0, 'last_embedding_s': -math.inf, 'samples': []})
                state['last_seen_s'] = timestamp
                state['count'] += 1
                obs = {'schema_version': 1, 'camera_id': camera['camera_id'],
                       'frame_index': frame_index, 'timestamp_s': timestamp,
                       'local_track_id': identity, 'bbox_xyxy': box,
                       'detection_confidence': confidence}
                if encoder and timestamp - state['last_embedding_s'] >= options.embedding_interval_s:
                    candidate = crop_quality(frame, box, [b for other, b, _ in records if other != identity], options)
                    if candidate:
                        crop, quality = candidate
                        pending.append((obs, state, crop, quality))
                # Keep frame order; appearance vectors are added below before serialization.
                state['_observation'] = obs
            # Crop copies are necessary: previews must not contaminate descriptors.
            vectors = encoder.encode([p[2] for p in pending]) if pending else []
            for (obs, state, _, quality), vector in zip(pending, vectors):
                obs['embedding'] = vector.tolist()
                obs['embedding_model_id'] = encoder.model_id
                state['last_embedding_s'] = timestamp
                state['samples'].append((quality, frame_index, vector))
                state['samples'].sort(key=lambda v: v[0], reverse=True)
                del state['samples'][options.top_k:]
            for identity, _, _ in records:
                obs = local[identity].pop('_observation')
                stream.write(json.dumps(obs, allow_nan=False) + '\n')
                count += 1
            if writer:
                for identity, box, _ in records:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(frame, f"{camera['camera_id']}:{identity}", (x1, max(20, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)
                writer.write(frame)
        if decoded == 0:
            raise ValueError(f'Video has no decodable frames: {video}')
        tracklets = []
        for identity, state in local.items():
            item = {'camera_id': camera['camera_id'], 'local_track_id': identity,
                    'start_s': state['start_s'], 'last_seen_s': state['last_seen_s'],
                    'observation_count': state['count'], 'embedding_sample_count': len(state['samples']),
                    'selected_frame_indices': [s[1] for s in state['samples']]}
            if state['samples']:
                item['embedding'] = normalized(np.mean([s[2] for s in state['samples']], axis=0)).tolist()
                item['embedding_model_id'] = encoder.model_id
            tracklets.append(item)
        summary = {'camera_id': camera['camera_id'], 'video_path': camera['video_path'],
                   'width': width, 'height': height, 'fps': fps,
                   'timestamp_method': 'frame_index/fps + time_offset_s (constant-frame-rate assumption)',
                   'decoded_frames': decoded, 'processed_frames': processed,
                   'preview_file': preview_path.name if writer else None,
                   'observation_count': count, 'wall_time_s': time.perf_counter() - start}
        return summary, tracklets, count
    finally:
        cap.release()
        if writer:
            writer.release()
