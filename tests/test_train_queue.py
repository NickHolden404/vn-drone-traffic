import json
from pathlib import Path

import pytest
import yaml

from cvt.train.job import keep_prev, train_kwargs
from cvt.train.queue import (
    PAUSED_EXIT,
    Ckpt,
    Job,
    PauseTraining,
    _subprocess_runner,
    check_data,
    data_sha256,
    job_state,
    load_queue,
    make_pause_callback,
    run_queue,
    seconds_per_epoch,
)


def _job(tmp_path: Path, name="j1", group="teachers", epochs=10) -> Job:
    (tmp_path / "m.pt").write_bytes(b"w")
    return Job(name=name, group=group, model=str(tmp_path / "m.pt"), data=str(tmp_path / "d.yaml"),
               project=str(tmp_path / "runs"), run=name, args={"epochs": epochs})


def _ckpt(job: Job, *names: str, content=b"x") -> None:
    job.wdir.mkdir(parents=True, exist_ok=True)
    for n in names:
        (job.wdir / n).write_bytes(content)


def _reader(states: dict[str, Ckpt | Exception]):
    """Fake read_ckpt keyed by file name; an Exception value is raised."""
    def read(p: Path) -> Ckpt:
        v = states[p.name]
        if isinstance(v, Exception):
            raise v
        return v
    return read


def _no_finalise(wdir: Path) -> None:
    raise AssertionError("finalise not expected")


def test_job_states(tmp_path: Path):
    j = _job(tmp_path)
    assert job_state(j, read=_reader({})) == ("todo", None)
    _ckpt(j, "last.pt")
    assert job_state(j, read=_reader({"last.pt": Ckpt(4, 10)}), finalise=_no_finalise) == ("paused", 5)
    assert job_state(j, read=_reader({"last.pt": Ckpt(-1, 10)})) == ("broken", None)
    _ckpt(j, "best.pt")
    assert job_state(j, read=_reader({"last.pt": Ckpt(-1, 10)})) == ("done", None)


def test_job_state_finalises_a_finished_unstripped_run(tmp_path: Path):
    j = _job(tmp_path)
    _ckpt(j, "last.pt", "best.pt")
    stripped = []
    state = job_state(j, read=_reader({"last.pt": Ckpt(9, 10)}), finalise=stripped.append, log=lambda m: None)
    assert state == ("done", None) and stripped == [j.wdir]


def test_job_state_restores_prev_when_last_is_torn(tmp_path: Path):
    j = _job(tmp_path)
    _ckpt(j, "last.pt", content=b"torn")
    _ckpt(j, "last.prev.pt", content=b"good")
    logs = []
    read = _reader({"last.pt": RuntimeError("PytorchStreamReader failed"), "last.prev.pt": Ckpt(3, 10)})
    assert job_state(j, read=read, log=logs.append) == ("paused", 4)
    assert (j.wdir / "last.pt").read_bytes() == b"good" and "restored last.prev.pt" in logs[0]
    assert not list(j.wdir.glob("*.tmp"))


def test_job_state_corrupt_when_both_fail(tmp_path: Path):
    j = _job(tmp_path)
    _ckpt(j, "last.pt")
    assert job_state(j, read=_reader({"last.pt": EOFError()})) == ("corrupt", None)  # no last.prev.pt
    _ckpt(j, "last.prev.pt")
    read = _reader({"last.pt": EOFError(), "last.prev.pt": RuntimeError("bad zip")})
    assert job_state(j, read=read) == ("corrupt", None)
    (tmp_path / "d.yaml").write_text("x")
    with pytest.raises(RuntimeError, match="last.prev.pt"):
        run_queue([j], "teachers", 1.0, runner=lambda jb, d: 0, read=read, clock=lambda: 0.0)


def test_keep_prev_copies_last(tmp_path: Path):
    last = tmp_path / "last.pt"
    last.write_bytes(b"epoch5")

    class T:
        pass

    t = T()
    t.last = last
    keep_prev(t)
    assert (tmp_path / "last.prev.pt").read_bytes() == b"epoch5" and not list(tmp_path.glob("*.tmp"))


def test_wdir_matches_ultralytics_save_dir(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("YOLO_OFFLINE", "1")
    cfg = pytest.importorskip("ultralytics.cfg")
    monkeypatch.chdir(tmp_path)
    job = Job(name="t", group="teachers", model="m.pt", data="d.yaml", project="runs/teachers", run="yolo11x",
              args={"epochs": 3, "imgsz": 320})
    args = cfg.get_cfg(cfg.DEFAULT_CFG, {**train_kwargs(job), "model": "m.pt", "mode": "train", "task": "detect"})
    assert cfg.get_save_dir(args) == job.wdir.parent
    assert job.wdir == tmp_path.resolve() / "runs" / "teachers" / "yolo11x" / "weights"


class FakeTrainer:
    def __init__(self, epoch_time, stop=False, epoch=3):
        self.epoch_time, self.stop, self.epoch = epoch_time, stop, epoch


def test_pause_callback_stops_before_overrunning():
    cb = make_pause_callback(deadline=1000.0, clock=lambda: 900.0)
    cb(FakeTrainer(epoch_time=50))  # 900 + 55 < 1000: keep going
    with pytest.raises(PauseTraining, match="after epoch 4"):
        cb(FakeTrainer(epoch_time=100))  # 900 + 110 > 1000: pause now, last.pt is saved
    cb(FakeTrainer(epoch_time=500, stop=True))  # finishing anyway: let Ultralytics finalise


def test_load_queue_rejects_duplicates(tmp_path: Path):
    q = tmp_path / "q.yaml"
    j = {"name": "a", "group": "g", "model": "m", "data": "d", "project": "p", "run": "a"}
    q.write_text(yaml.safe_dump({"jobs": [j, j]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_queue(q)


def test_shipped_queue_pins_student_optimizer():
    jobs = load_queue(Path("configs/train_queue.yaml"))
    for j in jobs:
        if j.group in ("students", "seeds", "smoke"):
            assert (j.args["optimizer"], j.args["lr0"], j.args["warmup_bias_lr"]) == ("AdamW", 0.001, 0.0), j.name
    assert {j.group for j in jobs} == {"teachers", "students", "seeds", "smoke"}


def test_run_queue_skips_done_stops_on_pause_and_checks_inputs(tmp_path: Path):
    (tmp_path / "d.yaml").write_text("x")
    done, paused, later = _job(tmp_path, "done"), _job(tmp_path, "p"), _job(tmp_path, "later")
    _ckpt(done, "last.pt", "best.pt")
    _ckpt(paused, "last.pt")
    states = {"done": Ckpt(-1, 10), "p": Ckpt(2, 10)}
    calls = []

    def read(p: Path) -> Ckpt:
        return states[p.parent.parent.name]

    def runner(job, deadline):
        calls.append(job.name)
        states[job.name] = Ckpt(4, 10)
        return PAUSED_EXIT

    out = run_queue([done, paused, later], "teachers", hours=1.0, runner=runner, read=read, clock=lambda: 0.0)
    assert calls == ["p"] and out == "paused"
    missing = _job(tmp_path, "x")
    missing.data = str(tmp_path / "nope.yaml")
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        run_queue([missing], "teachers", 1.0, runner=runner, read=read, clock=lambda: 0.0)
    fresh = _job(tmp_path, "f")
    fresh.model = str(tmp_path / "absent.pt")
    with pytest.raises(FileNotFoundError, match="absent.pt.*manually"):
        run_queue([fresh], "teachers", 1.0, runner=runner, read=read, clock=lambda: 0.0)


def test_run_queue_min_start(tmp_path: Path):
    (tmp_path / "d.yaml").write_text("x")
    j = _job(tmp_path)
    args = dict(runner=lambda jb, d: pytest.fail("must not start"), read=_reader({}), clock=lambda: 0.0)
    assert run_queue([j], "teachers", 0.001, **args) == "budget"  # 3.6 s < 600 s
    calls = []

    def runner(job, deadline):
        calls.append(job.name)
        _ckpt(job, "last.pt")
        return PAUSED_EXIT

    out = run_queue([j], "teachers", 0.001, runner=runner, read=lambda p: Ckpt(0, 10), clock=lambda: 0.0,
                    min_start=0)
    assert out == "paused" and calls == ["j1"]


def test_run_queue_checks_state_after_the_run(tmp_path: Path):
    (tmp_path / "d.yaml").write_text("x")
    j = _job(tmp_path)

    def runner(rc):
        def run(job, deadline):
            _ckpt(job, "last.pt")
            return rc
        return run

    unstripped = dict(read=lambda p: Ckpt(4, 10), clock=lambda: 0.0)
    with pytest.raises(RuntimeError, match="reported success but checkpoint is not finalised"):
        run_queue([j], "teachers", 1.0, runner=runner(0), **unstripped)
    (j.wdir / "last.pt").unlink()
    with pytest.raises(RuntimeError, match="exit code 75"):  # e.g. a crash that looks like a pause
        run_queue([j], "teachers", 1.0, runner=lambda jb, d: PAUSED_EXIT, read=_reader({}), clock=lambda: 0.0)


def test_run_queue_propagates_failures(tmp_path: Path):
    (tmp_path / "d.yaml").write_text("x")
    with pytest.raises(RuntimeError, match="exit code 1"):
        run_queue([_job(tmp_path)], "teachers", 1.0, runner=lambda j, d: 1, read=_reader({}), clock=lambda: 0.0)


def test_subprocess_runner_waits_for_child_on_ctrl_c(tmp_path: Path):
    class FakePopen:
        waits = 0

        def __init__(self, cmd):
            self.cmd = cmd

        def wait(self):
            FakePopen.waits += 1
            if FakePopen.waits == 1:
                raise KeyboardInterrupt
            return PAUSED_EXIT

    run = _subprocess_runner(tmp_path / "q.yaml", popen=FakePopen)
    with pytest.raises(KeyboardInterrupt):
        run(_job(tmp_path), 0.0)
    assert FakePopen.waits == 2  # waited for the child instead of abandoning it mid-save


def test_seconds_per_epoch_uses_latest_piece():
    assert seconds_per_epoch([]) is None
    assert seconds_per_epoch([100, 200, 300]) == 100
    assert seconds_per_epoch([100, 200, 300, 50, 110]) == 55  # second piece: 2 epochs in 110 s
    assert seconds_per_epoch([100, 200, 300, 80]) == 80


def test_check_data_records_and_detects_changes(tmp_path: Path):
    lst = tmp_path / "VD-train.txt"
    lst.write_text("a.jpg\n")
    d = tmp_path / "d.yaml"
    d.write_text(yaml.safe_dump({"train": [lst.as_posix()], "val": [lst.as_posix()], "names": {0: "car"}}))
    j = _job(tmp_path)
    check_data(j, fresh=True)
    meta = json.loads((j.wdir.parent / "queue_meta.json").read_text())
    assert meta["data_sha256"] == data_sha256(d)
    check_data(j, fresh=False)  # unchanged: fine
    lst.write_text("a.jpg\nb.jpg\n")
    with pytest.raises(RuntimeError, match="changed since the run started"):
        check_data(j, fresh=False)
    (j.wdir.parent / "queue_meta.json").unlink()
    with pytest.raises(RuntimeError, match="recorded None"):
        check_data(j, fresh=False)


def test_real_checkpoints_restore_and_finalise(tmp_path: Path, monkeypatch):
    """read_ckpt/finalise_ckpt on real torch files: a torn last.pt is replaced (no lingering mmap on
    Windows) and a saved-but-unstripped final epoch is finalised."""
    monkeypatch.setenv("YOLO_OFFLINE", "1")
    torch = pytest.importorskip("torch")
    pytest.importorskip("ultralytics.utils.torch_utils")
    from cvt.train.queue import read_ckpt

    j = _job(tmp_path)
    j.wdir.mkdir(parents=True)

    def save(name, epoch, epochs=10):
        ck = {"model": torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3)), "ema": None, "epoch": epoch,
              "optimizer": {"state": {}}, "train_args": {"epochs": epochs}, "train_results": {"x": [1]}}
        torch.save(ck, j.wdir / name)

    save("last.prev.pt", 3)
    save("last.pt", 4)
    raw = (j.wdir / "last.pt").read_bytes()
    (j.wdir / "last.pt").write_bytes(raw[: len(raw) // 2])  # torn write
    assert job_state(j, log=lambda m: None) == ("paused", 4)
    assert read_ckpt(j.wdir / "last.pt") == Ckpt(3, 10)
    save("last.pt", 9)
    save("best.pt", 7)
    assert job_state(j, log=lambda m: None) == ("done", None)
    assert read_ckpt(j.wdir / "last.pt").epoch == -1 and read_ckpt(j.wdir / "best.pt").epoch == -1
    assert not list(j.wdir.glob("*.tmp"))


def test_local_font_is_case_insensitive_and_never_downloads(tmp_path):
    from cvt.train.job import local_font

    win = tmp_path / "Fonts"
    win.mkdir()
    (win / "arial.ttf").write_bytes(b"x")
    assert local_font("Arial.ttf", [tmp_path / "missing", win]) == win / "arial.ttf"
    assert local_font("Arial.Unicode.ttf", [win]) is None
