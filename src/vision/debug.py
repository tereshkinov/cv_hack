"""One-file debugging entry point. Uses real inference, no hand-labelled IDs."""
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys


def choose_video():
    if sys.platform != 'darwin':
        raise ValueError('File chooser is available on macOS; pass a video path on this OS')
    result = subprocess.run(['osascript', '-e',
        'POSIX path of (choose file with prompt "Select video for tracking debug")'],
        text=True, capture_output=True)
    if result.returncode:
        raise ValueError('Video selection cancelled or unavailable')
    return Path(result.stdout.strip())


def debug_video(root, video=None, output=None, device='cpu', max_frames=None,
                open_result=False, choose=False, embedding='none', max_duration_s=180.0):
    from .pipeline import Options, run
    if choose:
        if video is not None:
            raise ValueError('Use a video path or --choose, not both')
        video = choose_video()
    if video is None:
        raise ValueError('Specify a video path or use --choose')
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f'Video not found: {video}')
    output = Path(output).resolve() if output else root / 'runs/debug' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    # Keep run() responsible for refusing existing output directories.
    configs = root / 'runs/debug-configs'
    configs.mkdir(parents=True, exist_ok=True)
    config = configs / (datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.json')
    config.write_text(json.dumps({'schema_version': 1, 'cameras': [
        {'camera_id': 'debug', 'video_path': str(video), 'time_offset_s': 0}]}))
    print(f'Input: {video}\nOutput: {output}\nProcessing video; preview opens after completion.', flush=True)
    report = run(config, output, root, Options(device=device, embedding=embedding,
                    max_frames=max_frames, max_duration_s=max_duration_s, preview=True))
    source = output / report['cameras'][0]['preview_file']
    preview = output / 'tracked.mp4'
    source.rename(preview)
    report['cameras'][0]['preview_file'] = preview.name
    (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    tracks = json.loads((output / 'tracklets.json').read_text())['tracklets']
    with (output / 'tracks.csv').open('w', newline='') as handle:
        fields = ['local_track_id', 'start_s', 'last_seen_s', 'observation_count']
        writer = csv.DictWriter(handle, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(tracks)
    note = (f'Local tracking debug\nVideo: {video}\nLocal tracks: {len(tracks)}\n'
            f'Observations: {report["observation_count"]}\n'
            f'Processing: {report["wall_time_s"]:.2f}s / {report["processed_fps"]:.2f} frames/s\n'
            f'Appearance backend: {embedding}\n\n'
            'LOCAL IDs only. Same color/ID means the local tracker retained a track.\n'
            'Trails are observed box-foot points in image pixels, not a calibrated floor map.\n'
            'Short/fragmented tracks may indicate detection or association errors.\n'
            'No invisible person is drawn during missed detections.\n'
            'This mode does not assign cross-camera IDs or apply the global 10s timeout.\n'
            'Default: appearance extraction off for faster visual debugging.\n')
    (output / 'README.txt').write_text(note)
    print(note + f'\nWatch: {preview}\nTable: {output / "tracks.csv"}', flush=True)
    if open_result:
        if sys.platform == 'darwin':
            subprocess.run(['open', str(preview)], check=False)
        else:
            print('Open tracked.mp4 in a video player.')
    return output
