"""Opt-in fallback to the next provider when a download is too slow."""

import itertools
from types import SimpleNamespace

import ffmpeg
import pytest

from aniworld.models import AniworldEpisode
from aniworld.models.common import common


@pytest.fixture(autouse=True)
def linux_download(monkeypatch):
    monkeypatch.setattr(common, "platform", SimpleNamespace(system=lambda: "Linux"))
    monkeypatch.setenv("ANIWORLD_NAMING_TEMPLATE", "{title} S{season}E{episode}.mkv")


def make_episode(tmp_path):
    return AniworldEpisode(
        "https://aniworld.to/anime/stream/example/staffel-1/episode-1",
        series=SimpleNamespace(
            title_cleaned="Example", release_year="2024", imdb="tt1"
        ),
        season=SimpleNamespace(season_number=1),
        episode_number=1,
        title_en="Title",
        selected_path=tmp_path,
        selected_language="German Sub",
    )


def test_watch_ignores_the_warmup_and_short_dips():
    watch = common._SlowDownloadWatch(2.0, start=0)
    assert not watch.too_slow("0.5x", 10)
    assert not watch.too_slow("1.5x", 40)
    assert not watch.too_slow("1.5x", 99)
    assert not watch.too_slow("3.0x", 100)
    assert not watch.too_slow("1.5x", 150)
    assert watch.too_slow("1.5x", 210)


@pytest.mark.parametrize("speed", ["N/A", ""])
def test_watch_ignores_unknown_speed(speed):
    watch = common._SlowDownloadWatch(2.0, start=0)
    assert not watch.too_slow(speed, 500)


@pytest.mark.parametrize(
    "enabled,has_fallback,min_speed,expected",
    [
        ("0", True, None, None),
        ("1", False, None, None),
        ("1", True, None, 2.0),
        ("1", True, "3.5", 3.5),
        ("1", True, "fast", 2.0),
    ],
)
def test_min_speed_only_applies_with_a_fallback(
    monkeypatch, enabled, has_fallback, min_speed, expected
):
    monkeypatch.setenv("ANIWORLD_SLOW_PROVIDER_FALLBACK", enabled)
    if min_speed is None:
        monkeypatch.delenv("ANIWORLD_SLOW_PROVIDER_MIN_SPEED", raising=False)
    else:
        monkeypatch.setenv("ANIWORLD_SLOW_PROVIDER_MIN_SPEED", min_speed)
    monkeypatch.setattr(
        common._slow_provider, "has_fallback", has_fallback, raising=False
    )
    assert common._slow_provider_min_speed() == expected


class SlowFFmpeg:
    """Stand-in for an ffmpeg process that keeps reporting the same speed."""

    def __init__(self, speed):
        lines = (
            f"frame={n} time=00:00:{n % 60:02d}.00 speed={speed}\r"
            for n in itertools.count(1)
        )
        self.stream = itertools.chain.from_iterable(line.encode() for line in lines)
        self.killed = False
        self.returncode = None
        self.stderr = self

    def read(self, _size):
        return b"" if self.killed else bytes([next(self.stream)])

    def kill(self):
        self.killed = True
        self.returncode = -9

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def slow_ffmpeg(monkeypatch):
    clock = itertools.count(0, 0.5)
    monkeypatch.setattr(common.time, "monotonic", lambda: next(clock))
    monkeypatch.setenv("ANIWORLD_SLOW_PROVIDER_FALLBACK", "1")
    monkeypatch.setattr(common._slow_provider, "has_fallback", True, raising=False)
    process = SlowFFmpeg("1.2x")
    monkeypatch.setattr(common.subprocess, "Popen", lambda *a, **k: process)
    return process


def test_slow_network_download_is_stopped(slow_ffmpeg, tmp_path):
    node = ffmpeg.input("https://cdn.example/master.m3u8").output(
        str(tmp_path / "a.mkv")
    )
    with pytest.raises(common.DownloadTooSlow):
        common._run_ffmpeg_with_progress(node)
    assert slow_ffmpeg.killed


def test_slow_local_remux_is_left_alone(slow_ffmpeg, tmp_path, monkeypatch):
    stop_after = itertools.count()
    monkeypatch.setattr(
        slow_ffmpeg,
        "read",
        lambda size: (
            b"" if next(stop_after) > 20000 else SlowFFmpeg.read(slow_ffmpeg, size)
        ),
    )
    slow_ffmpeg.returncode = 0
    node = ffmpeg.input(str(tmp_path / "in.mkv")).output(str(tmp_path / "out.mkv"))
    common._run_ffmpeg_with_progress(node)
    assert not slow_ffmpeg.killed


def test_too_slow_provider_falls_back_without_retrying(tmp_path, monkeypatch):
    episode = make_episode(tmp_path)
    monkeypatch.setattr(
        type(episode), "provider_attempt_order", lambda self: ("VOE", "Filemoon")
    )
    monkeypatch.setattr(type(episode), "stream_url", "https://cdn.example/master.m3u8")
    tried = []

    def download(stream_url, temp_full, *args):
        tried.append(episode.selected_provider)
        if episode.selected_provider == "VOE":
            raise common.DownloadTooSlow("stayed below 2.0x")
        temp_full.write_bytes(b"video")

    monkeypatch.setattr(common, "_download_full_stream", download)
    episode.download()
    assert tried == ["VOE", "Filemoon"]
    assert episode._episode_path.read_bytes() == b"video"
    assert common._slow_provider.has_fallback is False


def test_the_last_provider_is_never_marked_as_having_a_fallback(tmp_path, monkeypatch):
    episode = make_episode(tmp_path)
    monkeypatch.setattr(
        type(episode), "provider_attempt_order", lambda self: ("VOE", "Filemoon")
    )
    monkeypatch.setattr(type(episode), "stream_url", "https://cdn.example/master.m3u8")
    seen = []

    def download(stream_url, temp_full, *args):
        seen.append((episode.selected_provider, common._slow_provider.has_fallback))
        if episode.selected_provider == "VOE":
            raise RuntimeError("broken stream")
        temp_full.write_bytes(b"video")

    monkeypatch.setattr(common, "_download_full_stream", download)
    episode.download()
    assert seen[0] == ("VOE", True)
    assert seen[-1] == ("Filemoon", False)
