"""Launch the local CursBreaker server and open it in a browser."""

from __future__ import annotations

import argparse
import socket
import threading
import time
import webbrowser


def _wait_until_serving(host: str, port: int, timeout: float = 30.0) -> bool:
    """Block until the server accepts a TCP connection, or ``timeout`` passes.

    Returns True once it's listening. Opening the browser on a blind timer raced
    server startup -- on a slower machine the tab hit a dead port and never
    became usable -- so we wait for the port to actually answer first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def _open_browser_when_ready(url: str, host: str, port: int) -> None:
    """Open the app once the server is actually up. If that fails (no default
    browser, a headless/SSH session, an odd webbrowser setup), say so and show
    the URL so the app stays usable by hand -- the previous fire-and-forget
    ``webbrowser.open`` swallowed every failure, leaving a blank, silent app."""
    if not _wait_until_serving(host, port):
        return  # server never came up; the foreground error already explains why
    try:
        opened = webbrowser.open(url, new=2)
    except Exception:
        opened = False
    if not opened:
        print(
            "\nCouldn't open a browser automatically. "
            f"Open this address yourself:\n    {url}\n",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="cursbreaker", description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address")
    parser.add_argument("--port", type=int, default=8765, help="port")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="keep the server running after the browser tab closes",
    )
    args = parser.parse_args()

    import uvicorn
    from copy import deepcopy

    from .server import (
        install_access_log_filter,
        start_autoshutdown,
        sweep_stale_workspaces,
    )

    # Reclaim disk from any earlier run that exited without cleaning up: a crash
    # or OS kill leaves its temp workspace (copied uploads + outputs) behind.
    sweep_stale_workspaces()

    # A bind-all host isn't browseable; point the tab (and the printed URL) at
    # loopback instead.
    browse_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
    url = f"http://{browse_host}:{args.port}/"
    if not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, browse_host, args.port),
            daemon=True,
        ).start()

    if not args.keep_alive:
        start_autoshutdown()

    install_access_log_filter()

    # Swap uvicorn's default access formatter for our PrettyAccessFormatter
    # so successful requests show up as green "ok" lines instead of yellowish
    # "INFO" lines that read like warnings.
    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    log_config["formatters"]["access"] = {
        "()": "cursbreaker.server.PrettyAccessFormatter",
    }

    print(f"CursBreaker running at {url}  (Ctrl+C to stop)")
    uvicorn.run(
        "cursbreaker.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
