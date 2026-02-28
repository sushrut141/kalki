# Kalki Sandbox

This sandbox runs three local processes together so you can quickly try Kalki:

1. `kalkid` (gRPC database server)
2. FastAPI UI for manual store/query operations
3. Background row pusher that writes logs every 10ms

## Prerequisites

- `python3` with `venv`
- C++ build toolchain (same as main Kalki prerequisites)
- Local embedding model at `third_party/models/all-minilm-v2.gguf`

## Start Everything

From repo root:

```bash
./sandbox/startup.sh
```

This script will:

- create `sandbox/.venv`
- install Python deps
- build `kalkid`
- run `kalkid`, FastAPI UI, and the 10ms row pusher
- fail fast if the embedding model is missing
- delete previous sandbox data and logs before startup

## What You Get

Default local addresses:

- Kalki gRPC: `127.0.0.1:8080`
- statusz: [http://127.0.0.1:8081/statusz](http://127.0.0.1:8081/statusz)
- Sandbox UI: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Try It Out

1. Open the UI and use **Store Log** to add a manual conversation.
2. Use **Query Logs** with a natural-language question.
3. Open `/statusz` to see ingested counts, WAL records, fresh block count, and baked block table.
4. Check live background write activity in `sandbox/logs/pusher.log`.

## Stop

Press `Ctrl+C` in the terminal running `startup.sh`.

## Optional overrides

You can override ports/addresses:

```bash
KALKI_GRPC_ADDR=127.0.0.1:9090 \
KALKI_STATUSZ_ADDR=127.0.0.1:9091 \
SANDBOX_HTTP_ADDR=127.0.0.1:8090 \
./sandbox/startup.sh
```

To make ingestion progress visible sooner in `/statusz`, sandbox defaults
`--wal_read_batch_size=1` (override with `SANDBOX_WAL_READ_BATCH_SIZE`).

Compaction is configured to trigger every 100 records in the active fresh block.
Override with `SANDBOX_MAX_RECORDS_PER_FRESH_BLOCK`.

The background pusher stops automatically after pushing 500 records.
Override with `SANDBOX_PUSHER_MAX_INGESTED_RECORDS`.

Debug logging build (default):

- `startup.sh` builds `kalkid` in `Debug` mode into `build-debug`.
- Override if needed:

```bash
KALKI_BUILD_TYPE=RelWithDebInfo \
KALKI_BUILD_DIR=./build \
./sandbox/startup.sh
```
