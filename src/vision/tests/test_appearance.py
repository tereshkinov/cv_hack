import cv2
import numpy as np
import pytest

from src.vision.appearance_experiment import (
    color_evidence,
    cosine,
    evaluate,
    evaluate_ids,
    observation_variant,
    read_rows,
    reencode,
    suggestions,
)
from src.vision.appearance_preprocess import ShadowGamma
from src.vision.clothing_color import clothing_colors, clothing_similarity
from src.vision.embeddings import CorrectedAppearance, normalized
from src.vision.pipeline import Options, run
from src.vision.tests.test_pipeline import FakeTracker, configuration


class FakeEncoder:
    model_id = 'test:mean-blue-and-green:l2'
    dimension = 2

    def encode(self, crops):
        return [normalized(crop.mean(axis=(0, 1))[:2] + 1) for crop in crops]


@pytest.mark.parametrize('level', [0, 5, 150, 255])
def test_gamma_leaves_unreliable_extremes_and_ordinary_light_alone(level):
    source = np.full((64, 32, 3), level, dtype=np.uint8)
    out = ShadowGamma()(source)
    np.testing.assert_array_equal(out, source)
    assert not np.shares_memory(out, source)


def test_gamma_raises_luma_preserves_input_and_keeps_chroma_close():
    source = np.full((64, 32, 3), (64, 35, 20), dtype=np.uint8)
    original = source.copy()
    out = ShadowGamma(0.8)(source)
    before = cv2.cvtColor(source, cv2.COLOR_BGR2YCrCb).astype(int)
    after = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb).astype(int)
    assert np.all(after[:, :, 0] > before[:, :, 0])
    assert np.max(np.abs(after[:, :, 1:] - before[:, :, 1:])) <= 2
    assert out.shape == source.shape and out.dtype == np.uint8
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(out, ShadowGamma(0.8)(source))


def test_bad_gamma_and_crop_are_rejected():
    for gamma in [0, 1, float('nan')]:
        with pytest.raises(ValueError, match='gamma'):
            ShadowGamma(gamma)
    with pytest.raises(ValueError, match='dark luma'):
        ShadowGamma(dark_luma=255)
    with pytest.raises(ValueError, match='BGR uint8'):
        ShadowGamma()(np.zeros((12, 12, 3), dtype=float))


def test_gamma_has_one_policy_id_even_if_crops_are_unchanged():
    raw = FakeEncoder()
    corrected = CorrectedAppearance(raw, ShadowGamma())
    bright = np.full((64, 32, 3), 200, dtype=np.uint8)
    assert corrected.model_id != raw.model_id
    assert corrected.dimension == raw.dimension
    np.testing.assert_array_equal(corrected.encode([bright])[0], raw.encode([bright])[0])
    assert CorrectedAppearance(raw, ShadowGamma(0.7)).model_id != corrected.model_id


def test_color_discounts_black_and_blocked_regions():
    crop = np.full((100, 50, 3), (60, 100, 180), dtype=np.uint8)
    visible = np.ones(crop.shape[:2], dtype=bool)
    both = clothing_colors(crop, visible)
    assert set(both) == {'upper', 'lower'}
    assert clothing_similarity(both, both) == pytest.approx(1)
    assert clothing_colors(np.zeros_like(crop), visible) == {}
    upper_only = color_evidence(crop, (0, 0, 50, 100), [(0, 50, 50, 100)])
    assert set(upper_only) == {'upper'}
    assert clothing_similarity(upper_only, {'lower': both['lower']}) is None


def test_cached_experiment_preserves_observations_and_compares_spaces(tmp_path):
    config = configuration(tmp_path)
    first = {'camera_id': 'A', 'frame_index': 0, 'local_track_id': 'l1', 'timestamp_s': 0,
             'bbox_xyxy': [0, 4, 45, 65], 'embedding': [1, 0],
             'embedding_model_id': 'old', 'detection_confidence': 0.95}
    second = {**first, 'camera_id': 'B', 'frame_index': 2, 'timestamp_s': 0.5}
    missing = {**first, 'frame_index': 1, 'embedding': None}
    del missing['embedding_model_id']
    rows = [first, second, missing]
    cameras = [{'camera_id': 'A', 'video_path': 'test.avi', 'overlap_with': ['B']},
               {'camera_id': 'B', 'video_path': 'test.avi'}]
    features, models, colors, timing, decoded = reencode(
        cameras, rows, FakeEncoder(), ShadowGamma(), tmp_path, include_color=True,
    )
    assert config.exists() and decoded == 2 and len(features) == 2
    assert set(timing) == {'raw', 'gamma'}
    assert colors and len(set(models.values())) == 3
    a, b = ('A', 0, 'l1'), ('B', 2, 'l1')
    variant = list(observation_variant(rows, features, 'combined', models))
    assert variant[-1] == missing and first['embedding_model_id'] == 'old'
    assert len(variant[0]['embedding']) == 4
    assert np.linalg.norm(variant[0]['embedding']) == pytest.approx(1)
    assert cosine(features[a][2], features[b][2]) == pytest.approx(
        (cosine(features[a][0], features[b][0]) + cosine(features[a][1], features[b][1])) / 2
    )
    proposed = suggestions(rows, features, cameras, 1)
    assert proposed[0]['left'] == a and proposed[0]['right'] == b
    assert 'same_person' not in proposed[0]
    pairs = [{**proposed[0], 'split': 'holdout', 'same_person': True}]
    summary, scores = evaluate(pairs, features, {}, 0.72)
    assert summary['holdout']['raw']['positive_count'] == 1
    assert scores[0]['scores']['color'] is None
    metrics = evaluate_ids(pairs, [{**first, 'global_track_id': 'g1'}, {**second, 'global_track_id': 'g2'}])
    assert metrics['holdout']['split_pairs'] == 1
    assert metrics['holdout']['retained_id_pairs'] == 0


def test_pipeline_off_matches_old_encoder_and_gamma_preserves_detection_keys(tmp_path):
    config = configuration(tmp_path)
    for mode in ('off', 'gamma'):
        report = run(config, tmp_path / mode, tmp_path, Options(
            appearance_preprocess=mode, embedding_interval_s=0,
        ), FakeTracker, FakeEncoder())
        assert report['observation_count'] == 6
        expected = FakeEncoder.model_id
        if mode == 'gamma':
            expected = CorrectedAppearance(FakeEncoder(), ShadowGamma()).model_id
        assert report['embedding_model_id'] == expected
    before = read_rows(tmp_path / 'off' / 'observations.jsonl')
    after = read_rows(tmp_path / 'gamma' / 'observations.jsonl')
    assert all(row['embedding_model_id'] == FakeEncoder.model_id for row in before)
    keys = ('camera_id', 'frame_index', 'local_track_id', 'bbox_xyxy', 'timestamp_s', 'detection_confidence')
    assert [[r[k] for k in keys] for r in before] == [[r[k] for k in keys] for r in after]
    assert all(np.linalg.norm(row['embedding']) == pytest.approx(1) for row in after)


def test_return_in_new_episode_can_look_like_the_same_person():
    a, b = ('A', 0, 'l1'), ('B', 30, 'l2')
    pair = {'left': a, 'right': b, 'same_person': True, 'same_episode': False, 'split': 'holdout'}
    assignments = [
        {'camera_id': camera, 'frame_index': frame, 'local_track_id': local, 'global_track_id': gid}
        for (camera, frame, local), gid in [(a, 'g1'), (b, 'g2')]
    ]
    metrics = evaluate_ids([pair], assignments)['holdout']
    assert metrics['expected_different_id_pairs'] == 1
    assert metrics['split_pairs'] == metrics['false_merge_pairs'] == 0
