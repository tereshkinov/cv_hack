"""One offline job in an isolated process; status survives UI reloads."""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runs/.ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/.matplotlib"))


def update(folder, phase, message, status="running"):
    target = folder / "status.json"
    temp = folder / "status.tmp"
    temp.write_text(
        json.dumps(
            {
                "status": status,
                "phase": phase,
                "message": message,
                "updated_at": time.time(),
            },
            ensure_ascii=False,
        )
    )
    temp.replace(target)
    print(message, flush=True)


def execute(folder):
    started = time.perf_counter()
    spec = json.loads((folder / "job.json").read_text())

    def progress(phase, message):
        update(folder, phase, message)

    try:
        from src.vision.__main__ import prepare

        if spec["kind"] == "prepare":
            progress("prepare", "Проверяем и загружаем веса моделей")
            prepare(ROOT)
            update(folder, "done", "Модели готовы", "complete")
            return
        import torch

        from src.association.cli import main as associate
        from src.vision.pipeline import Options, run

        from .review_data import build_review

        torch.set_num_threads(4)
        torch.manual_seed(0)
        project = json.loads((folder / "project.json").read_text())
        cameras = json.loads((folder / "cameras.json").read_text())["cameras"]
        settings = spec["options"]
        progress("vision", "Находим людей, строим локальные треки и признаки внешности")
        report = run(
            folder / "cameras.json",
            folder / "vision",
            ROOT,
            Options(
                device=settings["device"],
                frame_stride=settings["frame_stride"],
                confidence=settings["confidence"],
                embedding="osnet",
            ),
        )
        progress("association", "Объединяем наблюдения между камерами")
        associate(
            [
                "--cameras",
                str(folder / "cameras.json"),
                "--observations",
                str(folder / "vision/observations.jsonl"),
                "--output-dir",
                str(folder / "association"),
                "--timeout-s",
                str(settings["timeout_s"]),
                "--similarity-threshold",
                str(settings["similarity_threshold"]),
                "--tracklet-threshold",
                str(settings["tracklet_threshold"]),
            ]
        )
        progress("preview", "Готовим синхронный просмотр и траектории")
        result = build_review(folder, project, cameras, settings, progress)
        message = f"Готово: {len(result['tracks'])} треков, {result['observation_count']} наблюдений"
        (folder / "report.md").write_text(f"""# Отчёт обработки
{message}

Время: {time.perf_counter() - started:.2f} с. Камер: {len(cameras)}.
Анализ: YOLO11n + ByteTrack + OSNet, собственное межкамерное сопоставление и офлайн-уточнение.
Порог первого прохода {settings["similarity_threshold"]}; нескольких кадров {settings["tracklet_threshold"]}; ожидание {settings["timeout_s"]} с.
Калиброваны {sum(c["calibrated"] for c in result["cameras"])} камер. Отброшено проекций: {result["projection_rejected"]}.
Предпросмотр {settings["preview_fps"]} кадров/с; анализ каждый {settings["frame_stride"]}-й исходный кадр.

## Ограничения и пример возможной ошибки
Одинаковая одежда, перекрытия, отражения и маленькие изображения могут вызвать склейки/разрывы и лишние треки.
При выходе и возврате раньше тайм-аута ID может сохраниться: это известное упрощение завершения.
Порог 0.65 был подобран на первых трёх роликах команды и требует проверки на новых людях.
Пример из начального прогона: по первым кадрам сходство между ракурсами было 0.71/0.66 при пороге 0.75, что разрывало основной трек. Офлайн-уточнение исправило его, но четыре дополнительных коротких трека остались.
Разметка пола должна соответствовать неподвижной камере; схема и высота сами по себе точной калибровки не дают.
Расходящиеся позиции нескольких камер не усредняются; отсутствующие наблюдения не дорисовываются.
Нет численной оценки точности без ручного эталона. Нужен новый независимый тест с несколькими людьми.

Версии и скорость: vision/run.json. Правила и лицензии: src/vision/THIRD_PARTY.md, src/association/README.md.
""")
        report["product_wall_time_s"] = time.perf_counter() - started
        (folder / "performance.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        update(folder, "done", message, "complete")
    except Exception as error:
        traceback.print_exc()
        update(folder, "failed", str(error), "failed")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    execute(parser.parse_args().run_dir.resolve())
