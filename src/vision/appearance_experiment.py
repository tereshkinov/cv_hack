"""Re-encode fixed detections and evaluate labeled cross-camera appearance pairs."""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from math import ceil, floor
from pathlib import Path
from statistics import mean
from time import perf_counter

import cv2
import numpy as np

from .appearance_preprocess import ShadowGamma
from .clothing_color import COLOR_MODEL_ID, clothing_colors, clothing_similarity
from .embeddings import OSNET_FILE, CorrectedAppearance, OSNet, normalized, sha256
from .pipeline import load_cameras, package_versions, write_json

ROOT = Path(__file__).resolve().parents[2]
MODES = ('raw', 'gamma', 'combined')


def reference(row):
    return row['camera_id'], row['frame_index'], row['local_track_id']


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_rows(path, rows):
    with path.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')


def pixel_box(row, width, height):
    x1, y1, x2, y2 = row['bbox_xyxy']
    return (max(0, floor(x1)), max(0, floor(y1)), min(width, ceil(x2)), min(height, ceil(y2)))


def color_evidence(crop, box, other_boxes):
    x1, y1, x2, y2 = box
    # A box from another detection may occlude the target. Mask that region conservatively.
    visible = np.ones(crop.shape[:2], dtype=bool)
    for a, b, c, d in other_boxes:
        left, top, right, bottom = max(x1, a), max(y1, b), min(x2, c), min(y2, d)
        if left < right and top < bottom:
            visible[top-y1:bottom-y1, left-x1:right-x1] = False
    return clothing_colors(crop, visible)


def reencode(cameras, rows, encoder, correction, root, include_color=False):
    """Keep all observation rows and the exact set of references with embeddings."""
    by_camera = {cam['camera_id']: defaultdict(list) for cam in cameras}
    if len({reference(row) for row in rows}) != len(rows):
        raise ValueError('duplicate observation reference')
    for row in rows:
        if row['camera_id'] not in by_camera:
            raise ValueError(f"unknown camera: {row['camera_id']}")
        if type(row['frame_index']) is not int or row['frame_index'] < 0:
            raise ValueError('frame index must be a nonnegative integer')
        by_camera[row['camera_id']][row['frame_index']].append(row)
    gamma_encoder = CorrectedAppearance(encoder, correction)
    model_ids = {
        'raw': encoder.model_id,
        'gamma': gamma_encoder.model_id,
        'combined': f'{gamma_encoder.model_id}:fusion=equal-cosine-concat-v1',
    }
    features, colors = {}, []
    encoding_s = dict.fromkeys(('raw', 'gamma'), 0.0)
    decoded_frames = 0
    for camera in cameras:
        frames = by_camera[camera['camera_id']]
        indices = sorted(idx for idx, batch in frames.items() if any(row.get('embedding') is not None for row in batch))
        if not indices:
            continue
        cap = cv2.VideoCapture(str(root / camera['video_path']))
        try:
            if not cap.isOpened():
                raise ValueError(f"cannot open video for {camera['camera_id']}")
            frame_index = -1
            for idx in indices:
                while frame_index < idx:
                    if not cap.grab():
                        raise ValueError(f"missing frame {idx} for {camera['camera_id']}")
                    frame_index += 1
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"cannot decode frame {idx} for {camera['camera_id']}")
                decoded_frames += 1
                height, width = frame.shape[:2]
                if width != camera.get('width', width) or height != camera.get('height', height):
                    raise ValueError(f"video dimensions changed for {camera['camera_id']}")
                batch = frames[idx]
                boxes = [pixel_box(row, width, height) for row in batch]
                selected, crops = [], []
                for i, (row, box) in enumerate(zip(batch, boxes, strict=True)):
                    if row.get('embedding') is None:
                        continue
                    x1, y1, x2, y2 = box
                    if x2 <= x1 or y2 <= y1:
                        raise ValueError(f'empty crop for {reference(row)}')
                    crop = frame[y1:y2, x1:x2].copy()
                    selected.append(row)
                    crops.append(crop)
                    if include_color:
                        evidence = color_evidence(crop, box, boxes[:i] + boxes[i+1:])
                        if evidence:
                            colors.append({'reference': reference(row), 'model_id': COLOR_MODEL_ID,
                                           'regions': evidence})
                start = perf_counter()
                raw = encoder.encode(crops)
                encoding_s['raw'] += perf_counter() - start
                start = perf_counter()
                gamma = gamma_encoder.encode(crops)
                encoding_s['gamma'] += perf_counter() - start
                for row, a, b in zip(selected, raw, gamma, strict=True):
                    # Cosine on the concatenation equals the mean of the two within-space cosines.
                    features[reference(row)] = (a, b, normalized(np.concatenate((a, b))))
        finally:
            cap.release()
    return features, model_ids, colors, encoding_s, decoded_frames


def observation_variant(rows, features, mode, model_ids):
    slot = MODES.index(mode)
    for row in rows:
        vectors = features.get(reference(row))
        if vectors is None:
            yield row
        else:
            yield {**row, 'embedding': vectors[slot].tolist(), 'embedding_model_id': model_ids[mode]}


def cosine(a, b):
    return float(np.clip(np.dot(a, b), -1, 1))


def suggestions(rows, features, cameras, limit):
    """Rank candidate pairs for manual review. No candidate is an identity label."""
    tracks = defaultdict(list)
    for row in rows:
        key = reference(row)
        if key in features:
            tracks[key[0], key[2]].append(row)
    overlaps = {frozenset((cam['camera_id'], other)) for cam in cameras
                for other in cam.get('overlap_with', [])}
    transitions = {(cam['camera_id'], other) for cam in cameras
                   for other in cam.get('transition_to', [])}
    candidates = []
    for track in tracks.values():
        track.sort(key=lambda row: row['timestamp_s'])
    for left_track, right_track in combinations(tracks.values(), 2):
        left_cam, right_cam = left_track[0]['camera_id'], right_track[0]['camera_id']
        if (frozenset((left_cam, right_cam)) not in overlaps
                and (left_cam, right_cam) not in transitions
                and (right_cam, left_cam) not in transitions):
            continue
        # Include the start and end of each track, not only its sharpest middle frame.
        representatives = [
            (track[0], max(track, key=lambda r: r['detection_confidence']), track[-1])
            for track in (left_track, right_track)
        ]
        alternatives = []
        for left in representatives[0]:
            for right in representatives[1]:
                a, b = reference(left), reference(right)
                t1, t2 = left['timestamp_s'], right['timestamp_s']
                earlier, later = (a, b) if t1 < t2 else (b, a)
                if abs(t1 - t2) > 15 or (
                    frozenset((a[0], b[0])) not in overlaps
                    and (earlier[0], later[0]) not in transitions
                ):
                    continue
                raw = cosine(features[a][0], features[b][0])
                gamma = cosine(features[a][1], features[b][1])
                priority = abs(raw - gamma) + 0.1 * max(raw, gamma)
                alternatives.append((priority, a, b, t1, t2, raw, gamma))
        if alternatives:
            candidates.append(max(alternatives))
    return [{'left': a, 'right': b, 'left_timestamp_s': t1, 'right_timestamp_s': t2,
             'raw': raw, 'gamma': gamma}
            for _, a, b, t1, t2, raw, gamma in sorted(candidates, reverse=True)[:limit]]


def evaluate(pairs, features, colors, threshold):
    by_split = defaultdict(list)
    seen = set()
    reference_splits = {}
    for row in pairs:
        a, b = tuple(row['left']), tuple(row['right'])
        key = frozenset((a, b))
        split, label = row['split'], row['same_person']
        if split not in {'calibration', 'holdout'} or type(label) is not bool:
            raise ValueError('pairs require split=calibration|holdout and a boolean same_person')
        episode = row.get('same_episode', label)
        if type(episode) is not bool or (episode and not label):
            raise ValueError('same_episode must be boolean and can only be true for the same person')
        if len(a) != 3 or len(b) != 3 or a[0] == b[0] or key in seen:
            raise ValueError('pairs must be unique and refer to two different cameras')
        if a not in features or b not in features:
            raise ValueError(f'pair lacks an eligible embedding: {a}, {b}')
        if any(reference_splits.setdefault(ref, split) != split for ref in (a, b)):
            raise ValueError('the same observation cannot appear in calibration and holdout')
        seen.add(key)
        scores = {mode: cosine(x, y)
                  for mode, x, y in zip(MODES, features[a], features[b], strict=True)}
        scores['color'] = clothing_similarity(colors.get(a, {}), colors.get(b, {}))
        by_split[split].append({**row, 'scores': scores})
    result = {}
    for split, rows in by_split.items():
        result[split] = {}
        for mode in (*MODES, 'color'):
            positive = [r['scores'][mode] for r in rows if r['same_person'] and r['scores'][mode] is not None]
            negative = [r['scores'][mode] for r in rows if not r['same_person'] and r['scores'][mode] is not None]
            result[split][mode] = {
                'positive_count': len(positive), 'negative_count': len(negative),
                'mean_positive': mean(positive) if positive else None,
                'mean_negative': mean(negative) if negative else None,
                'unscored_count': len(rows) - len(positive) - len(negative),
            }
            if mode != 'color':
                result[split][mode].update(
                    missed_pairs=sum(x < threshold for x in positive),
                    false_matches=sum(x >= threshold for x in negative),
                )
    return result, [row for rows in by_split.values() for row in rows]


def evaluate_ids(pairs, assignments):
    ids = {reference(row): row['global_track_id'] for row in assignments}
    result = defaultdict(lambda: {'expected_same_id_pairs': 0, 'expected_different_id_pairs': 0,
                                  'retained_id_pairs': 0, 'split_pairs': 0, 'false_merge_pairs': 0})
    for row in pairs:
        metrics = result[row['split']]
        matched = ids[tuple(row['left'])] == ids[tuple(row['right'])]
        if row.get('same_episode', row['same_person']):
            metrics['expected_same_id_pairs'] += 1
            metrics['retained_id_pairs' if matched else 'split_pairs'] += 1
        else:
            metrics['expected_different_id_pairs'] += 1
            metrics['false_merge_pairs'] += int(matched)
    return dict(result)


def run_association(args, pairs):
    from src.association.cli import main as associate

    metrics = {}
    for mode in MODES:
        folder = args.output / f'association-{mode}'
        command = [
            '--cameras', str(args.cameras), '--observations', str(args.output / f'{mode}.jsonl'),
            '--output-dir', str(folder), '--timeout-s', '15', '--similarity-threshold', '0.72',
            '--tracklet-threshold', '0.62', '--ambiguity-margin', '0.03', '--adjacency-bonus', '0.04',
        ]
        if args.project:
            command.extend(['--project', str(args.project)])
        started = perf_counter()
        associate(command)
        metrics[mode] = {
            'association_s': perf_counter() - started,
            'labeled_pair_metrics': evaluate_ids(pairs, read_rows(folder / 'assignments.jsonl')),
        }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', required=True, type=Path)
    parser.add_argument('--observations', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New directory')
    parser.add_argument('--pairs', type=Path, help='Optional labeled JSONL')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--weights', type=Path, default=ROOT / 'models' / OSNET_FILE)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--dark-luma', type=int, default=80)
    parser.add_argument('--pair-threshold', type=float, default=0.72)
    parser.add_argument('--color', action='store_true')
    parser.add_argument('--associate', action='store_true', help='Run the fixed tuning association profile')
    parser.add_argument('--project', type=Path, help='Use the same optional floor calibration for all modes')
    args = parser.parse_args()
    try:
        if not -1 <= args.pair_threshold <= 1:
            raise ValueError('pair threshold must be between -1 and 1')
        if args.project and not args.associate:
            raise ValueError('--project requires --associate')
        started = perf_counter()
        correction = ShadowGamma(args.gamma, args.dark_luma)
        cameras = load_cameras(args.cameras, ROOT)
        rows = read_rows(args.observations)
        pairs = read_rows(args.pairs) if args.pairs else []
        args.output.mkdir(parents=True, exist_ok=False)
        encoder = OSNet(args.weights, args.device)
        encoder.torch.set_num_threads(4)
        features, model_ids, colors, timings, decoded = reencode(
            cameras, rows, encoder, correction, ROOT, args.color
        )
        if not features:
            raise ValueError('source run contains no observations with embeddings')
        evidence = {tuple(row['reference']): row['regions'] for row in colors}
        summary, scores = evaluate(pairs, features, evidence, args.pair_threshold)
        for mode in MODES:
            write_rows(args.output / f'{mode}.jsonl', observation_variant(rows, features, mode, model_ids))
        if args.color:
            write_rows(args.output / 'clothing_color.jsonl', colors)
        if pairs:
            write_rows(args.output / 'pair_scores.jsonl', scores)
        else:
            write_rows(args.output / 'review_pairs.jsonl', suggestions(rows, features, cameras, 30))
        id_metrics = run_association(args, pairs) if args.associate else None
        write_json(args.output / 'experiment.json', {
            'source_observations_sha256': sha256(args.observations),
            'source_cameras_sha256': sha256(args.cameras),
            'labeled_pairs_sha256': sha256(args.pairs) if args.pairs else None,
            'models': model_ids, 'observations': len(rows), 'encoded_crops_per_mode': len(features),
            'color_evidence_count': len(colors), 'decoded_sample_frames': decoded,
            'device': args.device, 'packages': package_versions(),
            'encode_s': timings, 'wall_s': perf_counter() - started,
            'pair_threshold': args.pair_threshold, 'pair_metrics': summary,
            'association_metrics': id_metrics,
            'floor_project_sha256': sha256(args.project) if args.project else None,
        })
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
