"""Opt-in parallel episode downloads within one queue item."""

import json
import threading

import pytest

from aniworld.models.common.common import _ProgressBoard
from aniworld.web import db, worker


class FakeEpisode:
    def __init__(self, url, on_download):
        self.url = url
        self.on_download = on_download

    def download(self):
        self.on_download(self.url)


@pytest.fixture
def run_parallel(monkeypatch):
    monkeypatch.setenv("ANIWORLD_PARALLEL_DOWNLOADS", "1")

    def factory(queue_id, on_download):
        started = []

        def build(url, extra, item, selected_path):
            started.append(url)
            return None, FakeEpisode(url, on_download)

        monkeypatch.setattr(worker, "_build_episode", build)
        monkeypatch.setattr(worker, "_notify_discord", lambda item: None)
        worker._process(db.get_queue_item(queue_id))
        return started

    return factory


def urls(count):
    return [f"https://x/ep{n}" for n in range(1, count + 1)]


def test_two_episodes_download_at_the_same_time(queue_item, run_parallel):
    queue_id = queue_item(episodes=urls(5))
    db.set_queue_status(queue_id, "running")
    together = threading.Barrier(2, timeout=5)
    started = run_parallel(queue_id, lambda url: url in urls(2) and together.wait())
    assert sorted(started) == sorted(urls(5))
    item = db.get_queue_item(queue_id)
    assert item["status"] == "completed"
    assert item["current_episode"] == 5


def test_never_more_than_two_at_once(queue_item, run_parallel):
    queue_id = queue_item(episodes=urls(6))
    db.set_queue_status(queue_id, "running")
    lock, running, peak = threading.Lock(), [0], [0]

    def download(url):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        threading.Event().wait(0.05)
        with lock:
            running[0] -= 1

    run_parallel(queue_id, download)
    assert peak[0] == worker.PARALLEL_EPISODES


def test_episodes_run_one_by_one_when_disabled(queue_item, monkeypatch):
    monkeypatch.delenv("ANIWORLD_PARALLEL_DOWNLOADS", raising=False)
    assert worker._parallel_episodes() == 1


def test_a_failed_episode_is_recorded_and_the_rest_finish(queue_item, run_parallel):
    queue_id = queue_item(episodes=urls(4))
    db.set_queue_status(queue_id, "running")

    def download(url):
        if url.endswith("ep2"):
            raise RuntimeError("hoster down")

    started = run_parallel(queue_id, download)
    assert sorted(started) == sorted(urls(4))
    item = db.get_queue_item(queue_id)
    assert item["status"] == "completed"
    assert [error["url"] for error in json.loads(item["errors"])] == ["https://x/ep2"]


def test_only_an_all_round_failure_marks_the_item_failed(queue_item, run_parallel):
    queue_id = queue_item(episodes=urls(3))
    db.set_queue_status(queue_id, "running")

    def download(url):
        raise RuntimeError("hoster down")

    run_parallel(queue_id, download)
    assert db.get_queue_item(queue_id)["status"] == "failed"


def test_a_cancel_starts_nothing_new_but_lets_running_episodes_finish(
    queue_item, run_parallel
):
    queue_id = queue_item(episodes=urls(5))
    db.set_queue_status(queue_id, "running")
    together = threading.Barrier(2, timeout=5)
    cancelled = threading.Event()

    def download(url):
        together.wait()
        if url.endswith("ep1"):
            db.cancel_queue_item(queue_id)
            cancelled.set()
        else:
            assert cancelled.wait(5)

    started = run_parallel(queue_id, download)
    assert sorted(started) == urls(2)
    item = db.get_queue_item(queue_id)
    assert item["status"] == "cancelled"
    assert item["current_episode"] == 2


def test_a_force_cancel_counts_no_broken_episode_and_records_no_error(
    queue_item, run_parallel
):
    queue_id = queue_item(episodes=urls(4))
    db.set_queue_status(queue_id, "running")
    together = threading.Barrier(2, timeout=5)

    def download(url):
        together.wait()
        db.cancel_queue_item(queue_id, force=True)
        raise RuntimeError("ffmpeg killed")

    run_parallel(queue_id, download)
    item = db.get_queue_item(queue_id)
    assert item["status"] == "cancelled"
    assert item["current_episode"] == 0
    assert item["errors"] == "[]"


def test_progress_of_parallel_downloads_is_merged():
    board = _ProgressBoard()
    board.update(
        key=1,
        percent=20.0,
        time="00:01:00",
        speed="10x",
        bandwidth="2.0 MB/s",
        active=True,
    )
    board.update(
        key=2,
        percent=60.0,
        time="00:03:00",
        speed="12x",
        bandwidth="3.0 MB/s",
        active=True,
    )
    merged = board.snapshot()
    assert merged["active"] is True
    assert merged["percent"] == 80.0
    assert merged["bandwidth"] == "5.0 MB/s"
    assert merged["speed"] == "10x | 12x"

    board.update(key=1, active=False)
    assert board.snapshot()["percent"] == 60.0
    board.update(key=2, active=False)
    assert board.snapshot()["active"] is False


@pytest.mark.parametrize(
    "configured,expected",
    [(None, 2), ("3", 3), ("6", 6), ("1", 2), ("9", 6), ("many", 2)],
)
def test_the_number_of_episodes_at_once_is_configurable(
    monkeypatch, configured, expected
):
    monkeypatch.setenv("ANIWORLD_PARALLEL_DOWNLOADS", "1")
    if configured is None:
        monkeypatch.delenv("ANIWORLD_PARALLEL_EPISODES", raising=False)
    else:
        monkeypatch.setenv("ANIWORLD_PARALLEL_EPISODES", configured)
    assert worker._parallel_episodes() == expected


def test_three_episodes_run_at_once_when_configured(
    queue_item, run_parallel, monkeypatch
):
    monkeypatch.setenv("ANIWORLD_PARALLEL_EPISODES", "3")
    queue_id = queue_item(episodes=urls(6))
    db.set_queue_status(queue_id, "running")
    together = threading.Barrier(3, timeout=5)
    started = run_parallel(queue_id, lambda url: url in urls(3) and together.wait())
    assert sorted(started) == sorted(urls(6))
    assert db.get_queue_item(queue_id)["status"] == "completed"
