"""Algorithm-independent integration checks on generated videos, explicitly synthetic."""
import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

from src.vision.embeddings import ColorHistogram
from src.vision.pipeline import Options, clip_box, crop_quality, load_cameras, run


class Array:
    def __init__(self, x): self.x = np.asarray(x)
    def cpu(self): return self
    def numpy(self): return self.x


class FakeTracker:
    def __init__(self): self.calls = 0
    def track(self, frame, **kwargs):
        self.calls += 1
        # Simulate a missed detection; never invent a box for the missing frame.
        if self.calls == 2:
            return [SimpleNamespace(boxes=SimpleNamespace(id=None))]
        return [SimpleNamespace(boxes=SimpleNamespace(id=Array([1]),
                xyxy=Array([[-3, 4, 45, 65]]), conf=Array([0.91])))]


def make_video(path):
    rng = np.random.default_rng(7)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (80, 80))
    assert writer.isOpened()
    for _ in range(4):
        writer.write(rng.integers(0, 256, (80, 80, 3), dtype=np.uint8))
    writer.release()


def configuration(tmp_path):
    make_video(tmp_path / 'test.avi')
    path = tmp_path / 'cameras.json'
    path.write_text(json.dumps({'schema_version': 1, 'cameras': [
        {'camera_id': 'B', 'video_path': 'test.avi', 'time_offset_s': 0.1},
        {'camera_id': 'A', 'video_path': 'test.avi', 'time_offset_s': 0}]}))
    return path


def test_two_cameras_merge_times_coordinates_gaps_and_descriptors(tmp_path):
    config = configuration(tmp_path)
    instances = []
    def factory():
        tracker = FakeTracker(); instances.append(tracker); return tracker
    options = Options(embedding='color', embedding_interval_s=0, top_k=2, preview=True)
    report = run(config, tmp_path / 'out', tmp_path, options, factory)
    obs = [json.loads(x) for x in (tmp_path / 'out/observations.jsonl').read_text().splitlines()]
    assert len(instances) == 2 and all(t.calls == 4 for t in instances)
    assert len(obs) == 6 and report['observation_count'] == 6
    assert [o['timestamp_s'] for o in obs] == sorted(o['timestamp_s'] for o in obs)
    assert all(o['frame_index'] != 1 for o in obs)
    assert all(o['bbox_xyxy'] == [0, 4, 45, 65] for o in obs)
    assert all('global_track_id' not in o for o in obs)
    assert all(len(o['embedding']) == 384 and np.isclose(np.linalg.norm(o['embedding']), 1) for o in obs)
    assert all(o['embedding_model_id'].startswith('baseline:') for o in obs)
    tracks = json.loads((tmp_path / 'out/tracklets.json').read_text())['tracklets']
    assert len(tracks) == 2 and all(t['embedding_sample_count'] == 2 for t in tracks)
    assert {(t['camera_id'], t['local_track_id']) for t in tracks} == {('A', 'l1'), ('B', 'l1')}
    # Both local ID 1 values are isolated by camera; they are not asserted to be one person.
    with pytest.raises(FileExistsError):
        run(config, tmp_path / 'out', tmp_path, options, factory)


def test_quality_filters_and_no_aliasing():
    frame = np.random.default_rng(2).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    o = Options(min_crop_height=48)
    assert crop_quality(frame, [0, 0, 20, 20], [], o) is None
    assert crop_quality(frame, [0, 0, 60, 90], [[0, 0, 60, 90]], o) is None
    assert crop_quality(np.zeros_like(frame), [0, 0, 60, 90], [], o) is None
    crop, _ = crop_quality(frame, [0, 0, 60, 90], [], o)
    saved = crop.copy(); frame[:] = 0
    assert np.array_equal(crop, saved)
    assert clip_box([0, 0, float('nan'), 30], 100, 100) is None


def test_duplicate_camera_and_missing_files_fail_before_processing(tmp_path):
    config = configuration(tmp_path)
    data = json.loads(config.read_text())
    data['cameras'][1]['camera_id'] = 'B'
    config.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='unique'):
        load_cameras(config, tmp_path)
    data['cameras'][1]['camera_id'] = 'A'
    data['cameras'][1]['video_path'] = 'missing.mp4'
    config.write_text(json.dumps(data))
    with pytest.raises(FileNotFoundError):
        load_cameras(config, tmp_path)


def test_empty_detections_and_stride_are_not_fabricated(tmp_path):
    config = configuration(tmp_path)
    class Empty:
        def track(self, frame, **kw): return [SimpleNamespace(boxes=None)]
    report = run(config, tmp_path / 'out', tmp_path,
                 Options(embedding='none', frame_stride=2), Empty)
    assert report['observation_count'] == 0
    assert (tmp_path / 'out/observations.jsonl').read_text() == ''
    assert all(c['processed_frames'] == 2 for c in report['cameras'])
