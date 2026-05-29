"""Launch the local CurseBreaker server and open it in a browser."""

from __future__ import annotations

import argparse
import threading
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(prog="cursebreaker", description=__doc__)
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

    from .server import install_access_log_filter, start_autoshutdown

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    if not args.keep_alive:
        start_autoshutdown()

    install_access_log_filter()

    print(f"CurseBreaker running at {url}  (Ctrl+C to stop)")
    uvicorn.run("cursebreaker.server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
