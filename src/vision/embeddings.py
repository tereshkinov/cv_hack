"""Explicitly identified, L2-normalized appearance descriptors."""
from __future__ import annotations
import hashlib
from pathlib import Path
import cv2
import numpy as np

OSNET_FILE = 'osnet_x0_25_msmt17.pth'
OSNET_SHA256 = 'cf55163d78fc44c62c82f85ab62d39f10438679b5abe8c698ae08cfa84aa6e18'
OSNET_URL = ('https://huggingface.co/kaiyangzhou/osnet/resolve/'
             'a5c5cc037c24235cda3b21085b93ad77c9616224/'
             'osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_'
             'lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    if not np.isfinite(vector).all() or norm < 1e-12:
        raise ValueError('Appearance descriptor is non-finite or empty')
    return vector / norm


class ColorHistogram:
    """Baseline only: three vertical bands of H/S histograms, not learned ReID."""
    model_id = 'baseline:hsv-3bands-16x8:l2:v1'
    dimension = 384

    def encode(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        vectors = []
        for crop in crops:
            hsv = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2HSV)
            bands = np.array_split(hsv, 3, axis=0)
            hist = [cv2.calcHist([b], [0, 1], None, [16, 8], [0, 180, 0, 256]).ravel() for b in bands]
            vectors.append(normalized(np.concatenate([normalized(h) for h in hist])))
        return vectors


class OSNet:
    dimension = 512

    def __init__(self, weights: Path, device: str = 'cpu'):
        import torch
        from .vendor.osnet import osnet_x0_25
        if not weights.is_file():
            raise FileNotFoundError(f'OSNet weights missing: {weights}. Run python -m src.vision prepare')
        checksum = sha256(weights)
        if checksum != OSNET_SHA256:
            raise ValueError('OSNet weights differ from the verified MSMT17 checkpoint; refusing to mix models')
        self.model_id = f'osnet_x0_25:msmt17:sha256={checksum}:rgb256x128:imagenet:l2:v1'
        self.torch = torch
        self.device = torch.device(f'cuda:{device}' if str(device).isdigit() else device)
        self.model = osnet_x0_25(num_classes=1, pretrained=False)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        state = {k: v for k, v in state.items() if not k.startswith('classifier.')}
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if set(missing) != {'classifier.weight', 'classifier.bias'} or unexpected:
            raise ValueError(f'Incompatible checkpoint: missing={missing}, unexpected={unexpected}')
        self.model.to(self.device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]

    def encode(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        if not crops:
            return []
        arrays = [cv2.cvtColor(cv2.resize(c, (128, 256)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1) for c in crops]
        tensor = self.torch.from_numpy(np.stack(arrays)).float().to(self.device) / 255.0
        with self.torch.inference_mode():
            out = self.model((tensor - self.mean) / self.std).cpu().numpy()
        return [normalized(v) for v in out]


class CorrectedAppearance:
    """Apply one named preprocessing policy to every descriptor in a run."""

    def __init__(self, encoder, correction):
        self.encoder = encoder
        self.correction = correction
        self.dimension = encoder.dimension
        self.model_id = f'{encoder.model_id}:preprocess={correction.tag}'

    def encode(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        return self.encoder.encode([self.correction(crop) for crop in crops])
