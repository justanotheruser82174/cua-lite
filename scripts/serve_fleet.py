#!/usr/bin/env python3
"""Launcher for the cua-lite fleet router (FastAPI + uvicorn).

Routes one client-facing env-server endpoint across a fleet of env-server
nodes (started with [`scripts/serve_env.py`](/scripts/serve_env.py)). All
routing / health / merge logic lives in :mod:`lite.gym.remote.fleet`; this
file just builds config and hands off — mirror of ``scripts/serve_env.py``.

Usage::

    # nodes.txt: one env-server per line (host:port or full URL, # comments)
    printf 'envhost-1:30100\\nenvhost-2:30100\\n' > nodes.txt
    uv run python scripts/serve_fleet.py --port 30300 --nodes-file nodes.txt

    # clients point at the ROUTER exactly like a single env-server:
    CUA_LITE_ENV_SERVER_URL=http://routerhost:30300 \\
    CUA_LITE_ENV_SERVER_TOKEN=<their-token> ...

The nodes file is re-read on mtime change each poll — append a line to
hot-join a node. ``--node-token`` is the router's own bearer toward the
nodes (health/catalog polling); client bearers are forwarded verbatim.
"""

from __future__ import annotations

import argparse
import logging

import uvicorn

from lite.gym.remote.alive import (
    log_keep_alive_timeout,
    resolve_keep_alive_timeout,
)
from lite.gym.remote.fleet import FleetState, make_fleet_app, make_provider
from lite.utils.logging import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="cua-lite fleet router",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30300)
    parser.add_argument(
        "--provider",
        choices=["static"],
        default="static",
        help="Node roster source. 'static' reads --nodes-file.",
    )
    parser.add_argument(
        "--nodes-file",
        default=None,
        metavar="PATH",
        help=(
            "Static node roster, required when --provider static. One "
            "env-server per line; comments allowed; re-read on mtime change."
        ),
    )
    parser.add_argument(
        "--node-token",
        default="fleet-router",
        help=(
            "Bearer the router uses toward nodes for health/catalog polling. "
            "Client bearers are forwarded verbatim."
        ),
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=10.0,
        help="Seconds between node health polls (GET /host_status).",
    )
    parser.add_argument(
        "--catalog-interval",
        type=float,
        default=60.0,
        help="Seconds between env-catalog refreshes and /tasks cache TTL.",
    )
    parser.add_argument(
        "--timeout-keep-alive",
        type=float,
        default=None,
        dest="timeout_keep_alive",
        help=(
            "HTTP idle keep-alive timeout, seconds. Default is derived to "
            "stay above the client keep-alive window."
        ),
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Build fleet-router config and hand off to uvicorn."""
    setup_logging()
    logger = logging.getLogger("cua-lite.fleet-router")

    if args.poll_interval <= 0 or args.catalog_interval <= 0:
        raise SystemExit("--poll-interval and --catalog-interval must be > 0")
    provider = make_provider(args.provider, args.nodes_file)
    fleet = FleetState(
        provider,
        node_token=args.node_token,
        poll_interval_s=args.poll_interval,
        catalog_interval_s=args.catalog_interval,
    )
    logger.info(
        "fleet router: host=%s port=%d provider=%s nodes_file=%s "
        "poll=%.3gs catalog=%.3gs",
        args.host,
        args.port,
        args.provider,
        args.nodes_file,
        args.poll_interval,
        args.catalog_interval,
    )
    keep_alive = resolve_keep_alive_timeout(args.timeout_keep_alive)
    log_keep_alive_timeout(
        logger,
        keep_alive,
        explicit=args.timeout_keep_alive is not None,
    )
    uvicorn.run(
        make_fleet_app(fleet),
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=keep_alive,
    )


if __name__ == "__main__":
    main(_parse_args())
