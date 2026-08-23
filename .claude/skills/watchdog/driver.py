"""Concurrent SGLang throughput client.

Sends a fixed number of concurrent ``/generate`` requests to one or more local SGLang servers and
immediately reissues each one as it returns, keeping the servers' continuous-batching queues full.
Useful for warm-up and sustained throughput testing. ``ignore_eos`` makes every request decode the
full ``--max-new-tokens`` for steady, repeatable load. Pure stdlib + ``requests``.

Usage:
    python sglang_client.py --ports 31504 --concurrency 32
    python sglang_client.py --ports 31504,31505 --concurrency 32 --max-new-tokens 2048
"""

from __future__ import annotations

import argparse
import threading
import time

import requests

# Long-ish prompt so prefill does real work too; content is irrelevant.
PROMPT = "Write a long, detailed, and comprehensive essay about the history of computing. " * 8


def worker(url: str, max_new_tokens: int, stop: threading.Event) -> None:
    """Keep one request in flight at all times until stopped."""
    sess = requests.Session()
    payload = {
        "text": PROMPT,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 1.0,
            "ignore_eos": True,  # decode the full budget every time -> steady load
        },
    }
    while not stop.is_set():
        try:
            sess.post(url, json=payload, timeout=600)
        except Exception:
            # Server still booting or a transient hiccup — back off briefly and retry.
            time.sleep(1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", required=True, help="comma-separated SGLang ports to drive")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--concurrency", type=int, default=32,
                    help="in-flight requests per port (higher -> more load)")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args()

    stop = threading.Event()
    threads: list[threading.Thread] = []
    for port in args.ports.split(","):
        url = f"http://{args.host}:{port.strip()}/generate"
        for _ in range(args.concurrency):
            t = threading.Thread(target=worker, args=(url, args.max_new_tokens, stop), daemon=True)
            t.start()
            threads.append(t)

    print(f"sglang_client: {len(threads)} workers across ports {args.ports}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        stop.set()


if __name__ == "__main__":
    main()
