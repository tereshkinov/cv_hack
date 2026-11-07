"""Run from repository root: python -m src.vision --help."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Keep all incidental library caches inside ignored project output.
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'runs/.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs/.matplotlib'))


def prepare(root):
    import requests
    from .embeddings import OSNET_FILE, OSNET_SHA256, OSNET_URL, sha256
    target = root / 'models'
    target.mkdir(parents=True, exist_ok=True)
    files = [(OSNET_FILE, OSNET_URL, OSNET_SHA256),
             ('yolo11n.pt', 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt', None)]
    provenance = []
    for name, url, expected in files:
        path = target / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.part')
            try:
                print(f'Downloading {name}', flush=True)
                with requests.get(url, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as handle:
                        for block in response.iter_content(1024 * 1024):
                            handle.write(block)
                if expected and sha256(temporary) != expected:
                    raise ValueError(f'Checksum mismatch: {name}')
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        digest = sha256(path)
        if expected and digest != expected:
            raise ValueError(f'Checksum mismatch: {name}; remove wrong local weights and retry')
        provenance.append({'file': name, 'url': url, 'sha256': digest})
        print(f'{name}: sha256={digest}', flush=True)
    (target / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Agent 1: local-camera tracking and appearance observations')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('prepare', help='Download detector and verified person-ReID weights to models/')
    inspect = sub.add_parser('inspect', help='Report actual video metadata, without ML inference')
    inspect.add_argument('videos', nargs='+', type=Path)
    process = sub.add_parser('run', help='Process all configured cameras; no global ID assignment')
    process.add_argument('--cameras', type=Path, required=True)
    process.add_argument('--output', type=Path, required=True, help='New directory; existing directories are refused')
    process.add_argument('--detector', default='models/yolo11n.pt')
    process.add_argument('--device', default='cpu', help='cpu, mps or CUDA device (e.g. 0)')
    process.add_argument('--embedding', choices=['osnet', 'color', 'none'], default='osnet')
    process.add_argument('--reid-weights', default='models/osnet_x0_25_msmt17.pth')
    process.add_argument('--image-size', type=int, default=640)
    process.add_argument('--confidence', type=float, default=0.1)
    process.add_argument('--frame-stride', type=int, default=1)
    process.add_argument('--local-buffer-s', type=float, default=5.0)
    process.add_argument('--max-duration-s', type=float, default=180.0, help='First N seconds per camera; 0 processes the full recording')
    process.add_argument('--embedding-interval-s', type=float, default=0.5)
    process.add_argument('--min-crop-height', type=int, default=48)
    process.add_argument('--min-sharpness', type=float, default=10.0)
    process.add_argument('--max-overlap', type=float, default=0.6)
    process.add_argument('--top-k', type=int, default=5)
    process.add_argument('--max-frames', type=int)
    process.add_argument('--preview', action='store_true', help='Write diagnostic videos with LOCAL IDs')
    process.add_argument('--threads', type=int, default=4)
    debug = sub.add_parser('debug', help='Track one video and produce a watchable annotated MP4')
    debug.add_argument('video', type=Path, nargs='?')
    debug.add_argument('--choose', action='store_true', help='macOS file selection dialog')
    debug.add_argument('--output', type=Path)
    debug.add_argument('--open', dest='open_result', action='store_true', help='Open video player after processing')
    debug.add_argument('--device', default='cpu')
    debug.add_argument('--max-frames', type=int)
    debug.add_argument('--max-duration-s', type=float, default=180.0, help='First N seconds; 0 processes the full recording')
    debug.add_argument('--embedding', choices=['none', 'osnet', 'color'], default='none')
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            prepare(ROOT)
        elif args.command == 'debug':
            import torch
            from .debug import debug_video
            torch.set_num_threads(4)
            torch.manual_seed(0)
            debug_video(ROOT, args.video, args.output, args.device, args.max_frames,
                        args.open_result, args.choose, args.embedding, args.max_duration_s)
        elif args.command == 'inspect':
            import cv2
            for path in args.videos:
                cap = cv2.VideoCapture(str(path))
                try:
                    if not cap.isOpened():
                        raise ValueError(f'Cannot open {path}')
                    print(json.dumps({'path': str(path), 'fps': cap.get(cv2.CAP_PROP_FPS),
                        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}))
                finally:
                    cap.release()
        else:
            import torch
            from .pipeline import Options, run
            if args.threads < 1:
                raise ValueError('threads must be positive')
            torch.set_num_threads(args.threads)
            torch.manual_seed(0)
            options = Options(**{name: getattr(args, name) for name in Options.__dataclass_fields__})
            report = run(args.cameras.resolve(), args.output.resolve(), ROOT, options)
            print(json.dumps({'status': report['status'], 'observations': report['observation_count'],
                              'wall_time_s': report['wall_time_s'], 'processed_fps': report['processed_fps']}, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
