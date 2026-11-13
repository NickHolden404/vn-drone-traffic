"""Time-boxed, resumable training queue.

Each invocation trains the next unfinished jobs of a group for at most `hours`: a callback
stops training at the last epoch boundary that fits (last.pt is already saved there), and the
next invocation resumes from that epoch with the saved optimizer and LR schedule. Ultralytics'
own `time=` option is not used because it rescales epochs and the LR schedule."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from cvt.config import load_yaml

PAUSED_EXIT = 75  # EX_TEMPFAIL; 3 is what abort() returns on Windows
MIN_START_S = 600  # don't start or resume a job with less than 10 minutes left
CORRUPT = (RuntimeError, pickle.UnpicklingError, EOFError)  # what torch.load raises on a torn file


@dataclass
class Job:
    name: str
    group: str
    model: str
    data: str
    project: str
    run: str
    arch: str = "yolo"
    args: dict = field(default_factory=dict)

    @property
    def wdir(self) -> Path:
        """Absolute, like the project job.py hands to Ultralytics (a relative one lands under runs_dir/<task>/)."""
        return Path(self.project).resolve() / self.run / "weights"


@dataclass(frozen=True)
class Ckpt:
    epoch: int  # last completed epoch index; -1 once the optimizer is stripped (finalised)
    epochs: int  # planned epochs


class PauseTraining(Exception):
    pass


def ultralytics_offline() -> None:
    """Before importing Ultralytics: ONLINE=False (no DNS probe, telemetry, pip-update check), no pip installs."""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "0")


def load_queue(path: Path) -> list[Job]:
    jobs = [Job(**j) for j in load_yaml(path)["jobs"]]
    names = [j.name for j in jobs]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"duplicate job names in {path}: {dup}")
    return jobs


def read_ckpt(path: Path) -> Ckpt:
    """Epoch info of an Ultralytics checkpoint; raises one of CORRUPT on a torn file."""
    ultralytics_offline()  # unpickling imports ultralytics
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    try:
        stripped = ck.get("optimizer") is None
        return Ckpt(-1 if stripped else int(ck["epoch"]), int(ck["train_args"]["epochs"]))
    finally:
        del ck
        gc.collect()  # Windows: a live mapping would stop Ultralytics from rewriting last.pt


def copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def finalise_ckpt(wdir: Path) -> None:
    """Strip the optimizer from last.pt and best.pt, as Ultralytics' final_eval would have."""
    ultralytics_offline()
    from ultralytics.utils.torch_utils import strip_optimizer

    def strip(f: Path, **kw) -> dict:
        tmp = f.with_name(f.name + ".tmp")
        ck = strip_optimizer(f, s=str(tmp), **kw)
        if not ck:
            raise RuntimeError(f"could not finalise {f}")
        os.replace(tmp, f)
        return ck

    ck = strip(wdir / "last.pt")
    if (wdir / "best.pt").exists():
        strip(wdir / "best.pt", updates={"train_results": ck.get("train_results")})


def job_state(
    job: Job, read: Callable[[Path], Ckpt] = read_ckpt, finalise: Callable[[Path], None] = finalise_ckpt,
    log: Callable[[str], None] = print,
) -> tuple[str, int | None]:
    """("todo" | "paused" | "done" | "broken" | "corrupt", epochs completed when paused).
    A torn last.pt is replaced by last.prev.pt (the previous epoch); a run that died between its
    last save and the final strip is finalised here."""
    last, prev = job.wdir / "last.pt", job.wdir / "last.prev.pt"
    if not last.exists():
        return "todo", None
    try:
        ck = read(last)
    except CORRUPT as e:
        try:
            ck = read(prev) if prev.exists() else None
        except CORRUPT:
            ck = None
        if ck is None:
            return "corrupt", None
        copy_atomic(prev, last)
        log(f"{job.name}: {last} unreadable ({e}); restored {prev.name} (epoch {ck.epoch + 1})")
    if ck.epoch != -1 and ck.epoch + 1 >= ck.epochs:
        finalise(job.wdir)
        log(f"{job.name}: last epoch was saved but not finalised; stripped the optimizer")
        ck = Ckpt(-1, ck.epochs)
    if ck.epoch == -1:
        return ("done", None) if (job.wdir / "best.pt").exists() else ("broken", None)
    return "paused", ck.epoch + 1


def data_sha256(path: Path) -> str:
    """Hash of a dataset yaml and the .txt image lists it references (paths only, not label contents)."""
    h = hashlib.sha256(path.read_bytes())
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("train", "val", "test"):
        v = d.get(key) or []
        for p in v if isinstance(v, list) else [v]:
            if str(p).endswith(".txt"):
                h.update(Path(p).read_bytes())
    return h.hexdigest()


def check_data(job: Job, fresh: bool) -> None:
    """Record the dataset hash in <run>/queue_meta.json at a fresh start; refuse to resume if it changed."""
    meta, h = job.wdir.parent / "queue_meta.json", data_sha256(Path(job.data))
    if fresh:
        meta.parent.mkdir(parents=True, exist_ok=True)
        tmp = meta.with_name(meta.name + ".tmp")
        tmp.write_text(json.dumps({"data": job.data, "data_sha256": h}, indent=2), encoding="utf-8", newline="\n")
        os.replace(tmp, meta)
        return
    old = json.loads(meta.read_text(encoding="utf-8")).get("data_sha256") if meta.exists() else None
    if old != h:
        raise RuntimeError(f"{job.name}: {job.data} or its image lists changed since the run started "
                           f"(recorded {old}, now {h}); restore them or delete {job.wdir.parent} to restart")


def seconds_per_epoch(times: Sequence[float]) -> float | None:
    """Mean epoch time of the latest piece: results.csv's `time` restarts at 0 in every piece."""
    if not len(times):
        return None
    start = max((i for i in range(1, len(times)) if times[i] < times[i - 1]), default=0)
    piece = list(times[start:])
    return float(piece[-1]) / len(piece)


def make_pause_callback(deadline: float, clock: Callable[[], float] = time.time, margin: float = 1.1):
    """on_fit_epoch_end hook: pause when another epoch would end past `deadline`."""

    def on_fit_epoch_end(trainer) -> None:
        if trainer.stop:  # last epoch or early stop: let Ultralytics finalise the run
            return
        if clock() + margin * float(trainer.epoch_time or 0.0) > deadline:
            raise PauseTraining(f"paused after epoch {trainer.epoch + 1}")

    return on_fit_epoch_end


def _subprocess_runner(queue_file: Path, popen=subprocess.Popen) -> Callable[[Job, float], int]:
    def run(job: Job, deadline: float) -> int:
        cmd = [sys.executable, "-m", "cvt.train.job", "--queue", str(queue_file),
               "--name", job.name, "--deadline", f"{deadline:.0f}"]
        p = popen(cmd)
        try:
            return p.wait()
        except KeyboardInterrupt:  # the child got the same Ctrl+C; killing it mid-save would tear last.pt
            print("waiting for the training process to stop...", file=sys.stderr, flush=True)
            while True:
                try:
                    rc = p.wait()
                    break
                except KeyboardInterrupt:
                    continue
            print(f"{job.name}: training process exited with code {rc}", file=sys.stderr, flush=True)
            raise

    return run


def run_queue(
    jobs: Sequence[Job], group: str, hours: float, runner: Callable[[Job, float], int],
    read: Callable[[Path], Ckpt] = read_ckpt, finalise: Callable[[Path], None] = finalise_ckpt,
    clock: Callable[[], float] = time.time, log: Callable[[str], None] = print, min_start: float = MIN_START_S,
) -> str:
    """Train `group`'s unfinished jobs in order until done or out of time.
    Returns "done" (group finished), "paused" (budget used) or "budget" (no time to start)."""
    deadline = clock() + hours * 3600

    def state_of(job: Job) -> tuple[str, int | None]:
        return job_state(job, read, finalise, log)

    for job in [j for j in jobs if j.group == group]:
        state, epochs = state_of(job)
        if state == "done":
            continue
        last = job.wdir / "last.pt"
        if state == "broken":
            raise RuntimeError(f"{job.name}: {last} is finalised but best.pt is missing")
        if state == "corrupt":
            raise RuntimeError(f"{job.name}: neither {last} nor {last.with_name('last.prev.pt')} loads; "
                               f"delete {job.wdir.parent} to restart the job")
        if not Path(job.data).exists():
            raise FileNotFoundError(f"{job.name}: dataset yaml {job.data} missing")
        if state == "todo" and not Path(job.model).exists():
            raise FileNotFoundError(f"{job.name}: initial weights {job.model} missing; they must be downloaded "
                                    "manually")
        if clock() > deadline - min_start:
            log(f"time budget used; next: {job.name}")
            return "budget"
        log(f"=== {job.name}: {'resume after epoch ' + str(epochs) if state == 'paused' else 'start'}")
        rc = runner(job, deadline)
        if rc == PAUSED_EXIT:
            if (s := state_of(job)[0]) != "paused":
                raise RuntimeError(f"{job.name}: exit code {rc} (paused) but the checkpoint is {s}: {last}")
            log(f"{job.name}: paused at an epoch boundary; run again to resume")
            return "paused"
        if rc != 0:
            raise RuntimeError(f"{job.name}: training failed with exit code {rc}")
        if (s := state_of(job)[0]) != "done":
            raise RuntimeError(f"{job.name}: training reported success but checkpoint is not finalised "
                               f"({s}): {last}")
    return "done"
