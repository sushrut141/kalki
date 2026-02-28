from __future__ import annotations

import argparse
import random
import signal
import sys
import time

import grpc

from sandbox.kalki_grpc_client import KalkiGrpcClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="127.0.0.1:8080")
    parser.add_argument("--interval-ms", type=float, default=10.0)
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--agent-prefix", default="agent-pusher")
    parser.add_argument("--session-prefix", default="session-pusher")
    args = parser.parse_args()

    running = True

    def _stop(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    client = KalkiGrpcClient(args.target)
    sent = 0
    failures = 0

    try:
        while running:
            if args.max_records > 0 and sent >= args.max_records:
                print(
                    f"stop condition met: pushed={sent} max_records={args.max_records}",
                    flush=True,
                )
                break

            sequence_id = sent + 1
            agent_id = f"{args.agent_prefix}-{sequence_id}"
            session_id = f"{args.session_prefix}-{sequence_id}"
            conversation_log = (
                """Automated sandbox writer: collecting CI diagnostics and build history. "
                **Kalki** is a purpose-built database for the high-concurrency "Execution Era" of autonomous agents.
                While standard vector databases (Pinecone, Weaviate) are optimized for "finding similar sentences," 
                Kalki is a **Semantic Write-Ahead Log** designed to persist, compact, and retrieve millions of agent thought-chain 
                records without the "Decompression Tax."
                
                ## Motivation: The Context Crisis
                
                Autonomous agents running 24/7 produce millions of log lines. 
                Existing solutions (Standard RAG, Vector DBs, or Local Markdown files) force a trade-off:
                 - The Noise Problem: Indexing every raw "thought" token makes vector search imprecise.
                 - The Decompression Tax: To retrieve one specific context, current databases often have to decompress entire data pages, killing throughput.
                 - The Token Tax: Agents shouldn't have to read their entire history to find a single past decision.
                """
            )
            try:
                response = client.store_log(
                    agent_id=agent_id,
                    session_id=session_id,
                    conversation_log=conversation_log,
                )
                if response.status != 0:
                    failures += 1
                    if failures <= 10 or failures % 50 == 0:
                        print(
                            f"store_log non-OK status={response.status} "
                            f"error='{response.error_message}'",
                            flush=True,
                        )
            except grpc.RpcError as exc:
                failures += 1
                if failures <= 10 or failures % 50 == 0:
                    print(
                        f"store_log rpc error code={exc.code()} details={exc.details()}",
                        flush=True,
                    )

            sent += 1
            if sent % 100 == 0:
                print(f"sent={sent} failures={failures}", flush=True)
            time.sleep(max(args.interval_ms, 1.0) / 1000.0)
    finally:
        client.close()

    print(f"stopped sent={sent} failures={failures}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
