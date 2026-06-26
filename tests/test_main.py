"""Launcher behaviour: the browser is opened only once the server is actually
listening, and a failed open degrades to a clear, usable message instead of a
silent blank app (the "starts fine but no window ever opens" bug)."""

import socket

import cursbreaker.__main__ as m


def test_wait_until_serving_detects_a_live_port():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    try:
        assert m._wait_until_serving("127.0.0.1", port, timeout=2.0) is True
    finally:
        srv.close()


def test_wait_until_serving_times_out_on_a_dead_port():
    # Nothing is listening on port 1 -> never connects -> False within the timeout.
    assert m._wait_until_serving("127.0.0.1", 1, timeout=0.5) is False


def test_open_browser_waits_for_ready_then_opens(monkeypatch):
    monkeypatch.setattr(m, "_wait_until_serving", lambda *a, **k: True)
    opened = {}
    monkeypatch.setattr(m.webbrowser, "open", lambda url, **k: opened.setdefault("url", url) or True)
    m._open_browser_when_ready("http://127.0.0.1:9000/", "127.0.0.1", 9000)
    assert opened["url"] == "http://127.0.0.1:9000/"


def test_open_browser_prints_url_when_open_fails(monkeypatch, capsys):
    monkeypatch.setattr(m, "_wait_until_serving", lambda *a, **k: True)
    monkeypatch.setattr(m.webbrowser, "open", lambda *a, **k: False)  # no browser found
    m._open_browser_when_ready("http://127.0.0.1:8765/", "127.0.0.1", 8765)
    out = capsys.readouterr().out
    assert "Couldn't open a browser" in out and "http://127.0.0.1:8765/" in out


def test_open_browser_prints_url_when_open_raises(monkeypatch, capsys):
    monkeypatch.setattr(m, "_wait_until_serving", lambda *a, **k: True)

    def _boom(*a, **k):
        raise RuntimeError("no display")

    monkeypatch.setattr(m.webbrowser, "open", _boom)
    m._open_browser_when_ready("http://127.0.0.1:8765/", "127.0.0.1", 8765)
    assert "http://127.0.0.1:8765/" in capsys.readouterr().out


def test_open_browser_skips_opening_a_dead_server(monkeypatch):
    monkeypatch.setattr(m, "_wait_until_serving", lambda *a, **k: False)
    tried = {"open": False}
    monkeypatch.setattr(m.webbrowser, "open", lambda *a, **k: tried.__setitem__("open", True))
    m._open_browser_when_ready("http://127.0.0.1:8765/", "127.0.0.1", 8765)
    assert tried["open"] is False  # never opens a browser on a port that never came up
