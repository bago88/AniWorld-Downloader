"""Regressions for provider failures and preview URL construction."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from aniworld.extractors.common import ProviderUnavailable
from aniworld.extractors.provider import voe
from aniworld.models import AniworldEpisode
from aniworld.models.common import common
from aniworld.models.s_to import episode as sto
from aniworld.playwright import captcha


@pytest.fixture
def voe_preview_session(monkeypatch):
    session = SimpleNamespace(
        get=Mock(return_value=SimpleNamespace(text="", raise_for_status=Mock())),
        head=Mock(
            return_value=SimpleNamespace(
                headers={"Content-Type": "image/jpeg"}, raise_for_status=Mock()
            )
        ),
    )
    monkeypatch.setattr(voe, "GLOBAL_SESSION", session)
    return session


@pytest.mark.parametrize("quote", ["'", '"'])
@pytest.mark.parametrize("padding", ["", " \n\t"])
def test_voe_preview_requests_the_unquoted_url(voe_preview_session, quote, padding):
    session = voe_preview_session
    session.get.return_value.text = (
        f"window.location = {quote}{padding}https://mirror.example/e/abc{quote};"
    )
    headers = {"Referer": "https://voe.sx/"}
    expected = "https://mirror.example/cache/abc_storyboard_L2.jpg"

    result = voe.get_preview_image_link_from_voe(
        "https://voe.sx/e/start", headers=headers
    )

    assert result == expected
    session.get.assert_called_once_with("https://voe.sx/e/start", headers=headers)
    session.head.assert_called_once_with(
        expected, headers=headers, allow_redirects=True
    )
    session.get.return_value.raise_for_status.assert_called_once()
    session.head.return_value.raise_for_status.assert_called_once()


def test_voe_preview_without_a_redirect_does_not_request_an_image(voe_preview_session):
    with pytest.raises(ValueError, match="No redirect URL found"):
        voe.get_preview_image_link_from_voe("https://voe.sx/e/start")
    voe_preview_session.head.assert_not_called()


def test_voe_preview_rejects_a_non_image_response(voe_preview_session):
    voe_preview_session.get.return_value.text = (
        'window.location = "https://mirror.example/e/abc";'
    )
    voe_preview_session.head.return_value.headers = {"Content-Type": "text/html"}

    with pytest.raises(ValueError, match="Preview image not reachable"):
        voe.get_preview_image_link_from_voe("https://voe.sx/e/start")


@pytest.mark.parametrize("status", [404, 410])
@pytest.mark.parametrize("redirect", [False, True])
def test_deleted_voe_does_not_retry(monkeypatch, status, redirect):
    responses = [("deleted", "https://voe.sx/e/deleted", status)]
    if redirect:
        responses.insert(
            0,
            (
                'window.location="https://voe.sx/e/deleted"',
                "https://voe.sx/e/start",
                200,
            ),
        )
    fetch = Mock(side_effect=responses)
    sleep = Mock()
    monkeypatch.setattr(voe, "_voe_get", fetch)
    monkeypatch.setattr(voe.time, "sleep", sleep)
    with pytest.raises(ValueError, match=f"HTTP {status}"):
        voe.get_direct_link_from_voe("https://voe.sx/e/start")
    assert fetch.call_count == len(responses)
    sleep.assert_not_called()


def test_temporary_voe_failure_still_retries(monkeypatch):
    fetch = Mock(
        side_effect=[
            ("error", "https://voe.sx/e/start", 500),
            ("'hls': 'https://cdn.example/video.m3u8'", "https://voe.sx/e/start", 200),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(voe, "_voe_get", fetch)
    monkeypatch.setattr(voe.time, "sleep", sleep)
    assert (
        voe.get_direct_link_from_voe("https://voe.sx/e/start")
        == "https://cdn.example/video.m3u8"
    )
    sleep.assert_called_once_with(2)


BLOCKED = ("<html>403 Forbidden</html>", "https://voe.sx/e/start", 403)
VIDEO = ("'hls': 'https://cdn.example/video.m3u8'", "https://voe.sx/e/start", 200)


@pytest.fixture
def voe_static(monkeypatch):
    fetch = Mock()
    solve = Mock()
    sleep = Mock()
    monkeypatch.setattr(voe, "_voe_get", fetch)
    monkeypatch.setattr(voe, "solve_captcha", solve)
    monkeypatch.setattr(voe.time, "sleep", sleep)
    return SimpleNamespace(fetch=fetch, solve=solve, sleep=sleep)


@pytest.mark.parametrize(
    "solved,responses",
    [(None, [BLOCKED]), ("https://voe.sx/e/start", [BLOCKED, BLOCKED])],
    ids=["timed-out", "still-blocked"],
)
def test_unsolved_voe_challenge_opens_the_browser_only_once(
    voe_static, solved, responses
):
    voe_static.fetch.side_effect = responses
    voe_static.solve.return_value = solved
    with pytest.raises(ProviderUnavailable, match="VOE"):
        voe.get_direct_link_from_voe("https://voe.sx/e/start")
    voe_static.solve.assert_called_once_with("https://voe.sx/e/start")
    assert voe_static.fetch.call_count == len(responses)
    voe_static.sleep.assert_not_called()


def test_unsolved_challenge_on_the_voe_redirect_does_not_retry(voe_static):
    redirect = (
        'window.location="https://voe.sx/e/mirror"',
        "https://voe.sx/e/start",
        200,
    )
    voe_static.fetch.side_effect = [redirect, BLOCKED]
    voe_static.solve.return_value = None
    with pytest.raises(ProviderUnavailable, match="not solved"):
        voe.get_direct_link_from_voe("https://voe.sx/e/start")
    voe_static.solve.assert_called_once_with("https://voe.sx/e/mirror")
    voe_static.sleep.assert_not_called()


def test_solved_voe_challenge_still_resolves_the_video(voe_static):
    voe_static.fetch.side_effect = [BLOCKED, VIDEO]
    voe_static.solve.return_value = "https://voe.sx/e/start"
    assert (
        voe.get_direct_link_from_voe("https://voe.sx/e/start")
        == "https://cdn.example/video.m3u8"
    )
    voe_static.solve.assert_called_once()


def test_unavailable_provider_falls_back_without_retrying(tmp_path, monkeypatch):
    episode = AniworldEpisode(
        "https://aniworld.to/anime/stream/example/staffel-1/episode-1",
        series=SimpleNamespace(
            title_cleaned="Example", release_year="2024", imdb="tt123"
        ),
        season=SimpleNamespace(season_number=1),
        episode_number=1,
        title_de="Titel",
        title_en="Title",
        selected_path=tmp_path,
        selected_language="German Dub",
    )
    monkeypatch.setattr(common, "platform", SimpleNamespace(system=lambda: "Linux"))
    monkeypatch.setattr(
        common,
        "check_downloaded",
        Mock(
            return_value={"exists": False, "audio_langs": set(), "video_langs": set()}
        ),
    )
    monkeypatch.setattr(
        type(episode), "provider_attempt_order", lambda self: ("VOE", "Vidoza")
    )
    attempts = []

    def stream(self):
        attempts.append(self.selected_provider)
        if self.selected_provider == "VOE":
            raise ProviderUnavailable("VOE challenge was not solved")
        raise ValueError("Vidoza is down")

    monkeypatch.setattr(type(episode), "stream_url", property(stream))
    with pytest.raises(RuntimeError, match="VOE"):
        episode.download()
    assert attempts == ["VOE", "Vidoza", "Vidoza", "Vidoza"]


def test_missing_chromium_error_survives_modal_solver(monkeypatch, tmp_path):
    from patchright import sync_api

    from aniworld import autodeps

    monkeypatch.setenv("ANIWORLD_NO_AUTO_INSTALL", "1")
    runtime = MagicMock()
    runtime.chromium.executable_path = str(tmp_path / "missing-chromium")
    playwright = MagicMock()
    playwright.return_value.__enter__.return_value = runtime
    monkeypatch.setattr(sync_api, "sync_playwright", playwright)
    monkeypatch.setattr(autodeps, "_ensure_xvfb", Mock())
    with pytest.raises(RuntimeError, match="python -m patchright install chromium"):
        captcha.solve_sto_modal(
            "https://serienstream.to/serie/example/staffel-1/episode-1",
            "VOE",
            "Deutsch",
        )
    runtime.chromium.launch.assert_not_called()
    runtime.chromium.launch_persistent_context.assert_not_called()


def test_failed_modal_does_not_cache_serienstream_url(monkeypatch):
    episode = sto.SerienstreamEpisode(
        "https://serienstream.to/serie/example/staffel-1/episode-1"
    )
    redirect = "https://serienstream.to/r?t=example"
    monkeypatch.setattr(sto.SerienstreamEpisode, "provider_link", lambda *a: redirect)
    monkeypatch.setattr(sto, "sto_get", lambda *a: SimpleNamespace(url=redirect))
    solve = Mock(return_value=None)
    monkeypatch.setattr(captcha, "solve_sto_modal", solve)
    for _ in range(2):
        with pytest.raises(ValueError, match="Failed to resolve provider URL"):
            _ = episode.stream_url
    assert solve.call_count == 2
