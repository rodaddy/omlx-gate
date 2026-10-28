"""Start oMLX with the admission gate attached.

This is the plugin attach point. oMLX has no plugin system, but its FastAPI
`app` is a module-level object, and `serve_command` does real setup before
uvicorn (settings init, logging, burst-decode env, process title, socket
pre-bind) that we must not reimplement. So this does not replace
`omlx serve`; it imports the app, adds our middleware, then hands argv to
oMLX's own CLI untouched.

    .venv/bin/python -m omlx_gate.launch serve --model-dir ... --host ...

Every `omlx serve` flag passes through unchanged. Starlette builds the
middleware stack on first request, so add_middleware after import and
before main() is the correct ordering (proven by tests/test_launch.py).
"""

from __future__ import annotations

import sys


def main() -> None:
    # Import order matters: the app must exist before we attach to it, and
    # the gate import must not happen before omlx.server finishes building
    # the app (pool.py's lazy import guards the reverse direction).
    from omlx import server as omlx_server
    from omlx_gate.middleware import BigModelGateMiddleware

    omlx_server.app.add_middleware(BigModelGateMiddleware)

    from omlx.cli import main as omlx_main

    sys.exit(omlx_main())


if __name__ == "__main__":
    main()
