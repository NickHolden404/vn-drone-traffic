"""Run one queue job (fresh or resumed) with a pause deadline; exit PAUSED_EXIT (75) when paused."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cvt.train.queue import (
    PAUSED_EXIT,
    Job,
    PauseTraining,
    check_data,
    copy_atomic,
    job_state,
    load_queue,
    make_pause_callback,
    ultralytics_offline,
)


def train_kwargs(job: Job) -> dict:
    """model.train() arguments of a fresh run. The project is absolute: Ultralytics puts a relative
    one under SETTINGS["runs_dir"]/<task>/, where job.wdir would not find it."""
    return {"data": job.data, "project": str(Path(job.project).resolve()), "name": job.run,
            "exist_ok": True, **job.args}


def keep_prev(trainer) -> None:
    """on_model_save (fires after last.pt is fully written): keep a copy, so a torn next save can be undone."""
    copy_atomic(trainer.last, trainer.last.with_name("last.prev.pt"))


def local_font(font: str, dirs: list[Path]) -> Path | None:
    """First file named like `font` (case-insensitive: Windows ships arial.ttf) in `dirs`; never downloads."""
    name = Path(font).name.lower()
    for d in dirs:
        if d.is_dir():
            for f in d.iterdir():
                if f.name.lower() == name:
                    return f
    return None


def no_network() -> None:
    """In-process guards on top of ultralytics_offline(); settings.json is not touched."""
    import os

    import ultralytics.data.utils as data_utils
    import ultralytics.engine.trainer as trainer
    import ultralytics.utils.downloads as downloads
    import ultralytics.utils.plotting as plotting
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR

    dict.__setitem__(SETTINGS, "sync", False)  # bypasses SettingsManager's save: in memory only
    # RTX 4070 is not on Ultralytics' AMP-problem list; skipping check_amp avoids downloading yolo26n.pt.
    trainer.check_amp = lambda model: True
    # check_font downloads Arial.ttf when it is not in USER_CONFIG_DIR (its system-font lookup is
    # case-sensitive and misses Windows' arial.ttf). Fonts only label plots; None -> PIL default font.
    font_dirs = [USER_CONFIG_DIR, Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"]
    data_utils.check_font = plotting.check_font = lambda font="Arial.ttf": local_font(font, font_dirs)

    def refuse(url="", *args, **kwargs):
        raise RuntimeError(f"Ultralytics tried to download {url}; fetch it manually instead")

    downloads.safe_download = data_utils.safe_download = refuse


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--deadline", type=float, required=True)
    a = ap.parse_args(argv)
    job = {j.name: j for j in load_queue(a.queue)}[a.name]
    state, _ = job_state(job)
    if state == "done":
        return 0
    if state not in ("todo", "paused"):
        raise RuntimeError(f"{job.name}: checkpoint state is {state}; see `cvt train run`")
    resume = state == "paused"
    if not resume and not Path(job.model).exists():
        raise FileNotFoundError(f"{job.name}: initial weights {job.model} missing; they must be downloaded "
                                "manually")
    check_data(job, fresh=not resume)
    ultralytics_offline()
    from ultralytics import RTDETR, YOLO

    no_network()
    cls = RTDETR if job.arch == "rtdetr" else YOLO
    model = cls(str(job.wdir / "last.pt") if resume else job.model)
    model.add_callback("on_fit_epoch_end", make_pause_callback(a.deadline))
    model.add_callback("on_model_save", keep_prev)
    try:
        if resume:
            model.train(resume=True)
        else:
            model.train(**train_kwargs(job))
    except PauseTraining as e:
        trainer = getattr(model, "trainer", None)
        for loader in (getattr(trainer, "train_loader", None), getattr(trainer, "test_loader", None)):
            if callable(getattr(loader, "close", None)):
                loader.close()  # join persistent dataloader workers before exiting
        print(f"{job.name}: {e}", flush=True)
        return PAUSED_EXIT
    return 0


if __name__ == "__main__":
    sys.exit(main())
