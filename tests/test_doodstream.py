"""Doodstream extraction through its Cloudflare-gated mirrors."""

from types import SimpleNamespace

import pytest

from aniworld.extractors.provider import doodstream

EMBED_HTML = """
<script>
$.get('/pass_md5/123-45-abc/xyz', function(data) {
    window.location = data + makePlay() + '?token=t0k3n&expiry=' + Date.now();
});
</script>
"""


@pytest.fixture
def requests_seen(monkeypatch):
    seen = []

    def get(url, **kwargs):
        seen.append((url, kwargs))
        if "/pass_md5/" in url:
            return _response(url, "https://cdn.example/video~")
        return _response("https://playmogo.com/e/abc123", EMBED_HTML)

    monkeypatch.setattr(doodstream.requests, "get", get)
    return seen


def _response(url, text):
    return SimpleNamespace(url=url, text=text, raise_for_status=lambda: None)


def test_doodstream_requests_use_chrome_tls_with_verification_and_timeout(
    requests_seen,
):
    doodstream.get_direct_link_from_doodstream("https://dood.li/e/abc123")

    assert len(requests_seen) == 2
    for _url, kwargs in requests_seen:
        assert kwargs["impersonate"] == "chrome"
        assert kwargs["timeout"] == doodstream.REQUEST_TIMEOUT
        assert kwargs.get("verify", True) is True


def test_pass_md5_is_requested_from_the_mirror_the_embed_redirected_to(
    requests_seen,
):
    doodstream.get_direct_link_from_doodstream("https://do7go.com/e/abc123")

    pass_md5_url, kwargs = requests_seen[1]
    assert pass_md5_url == "https://playmogo.com/pass_md5/123-45-abc/xyz"
    assert kwargs["headers"]["Referer"] == "https://playmogo.com/e/abc123"


def test_direct_link_appends_random_suffix_token_and_expiry(requests_seen):
    link = doodstream.get_direct_link_from_doodstream("https://dood.li/e/abc123")

    base, query = link.split("?")
    assert base.startswith("https://cdn.example/video~")
    assert len(base) == len("https://cdn.example/video~") + 10
    assert query.startswith("token=t0k3n&expiry=")
