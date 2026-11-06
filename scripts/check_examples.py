"""Проверяет согласованность синтетических примеров, не алгоритм."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((root / 'examples' / name).read_text())
def lines(name):
    return [json.loads(s) for s in (root / 'examples' / name).read_text().splitlines() if s.strip()]
cameras = read('cameras.json')
tracks = read('tracks.json')
observations = lines('observations.jsonl')
assignments = lines('assignments.jsonl')
assert cameras['synthetic'] and tracks['synthetic']
cam_ids = {c['camera_id'] for c in cameras['cameras']}
track_ids = {t['global_track_id'] for t in tracks['tracks']}
assert len(track_ids) == len(tracks['tracks'])
keys = set()
for o in observations:
    assert o['schema_version'] == 1 and o['camera_id'] in cam_ids
    assert o['timestamp_s'] >= 0 and o['frame_index'] >= 0
    x1, y1, x2, y2 = o['bbox_xyxy']
    assert 0 <= x1 < x2 and 0 <= y1 < y2
    keys.add((o['camera_id'], o['frame_index'], o['local_track_id']))
for a in assignments:
    assert (a['camera_id'], a['frame_index'], a['local_track_id']) in keys
    assert a['global_track_id'] in track_ids
assert len(assignments) == len(observations)
for t in tracks['tracks']:
    assert t['start_s'] <= t['last_seen_s'] == t['end_s']
    assert set(t['camera_ids']) <= cam_ids
print('OK: synthetic contract examples are consistent; tracking is not implemented.')
