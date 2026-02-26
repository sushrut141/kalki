# Kalki

Single-instance leader database for autonomous-agent conversation logs.

## What is implemented

- gRPC API generated from protobuf definitions:
  - `StoreLog(agent_id, session_id, conversation_log, timestamp)`
  - `QueryLogs(caller_agent_id, query, optional filters...)`
- WAL-first write path:
  - Appends protobuf `WalRecord` to WAL with length prefix
  - Flushes to disk before returning success
- Ingestion worker:
  - Tracks WAL progress in SQLite metadata DB
  - Summarizes conversation logs with inheritable LLM interface (`LlmClient`)
  - Gemini implementation (`GeminiLlmClient`) calls Gemini `generateContent` over HTTPS
  - Boundary-aware splitting is delegated to Gemini via prompt rules for >500-word logs
- Block storage:
  - Fresh blocks: uncompressed protobuf list
  - Baked blocks: sorted by `(timestamp, agent_id, session_id)`
  - Header is compressed as a whole, data records individually compressed (zstd if available)
  - Header stores timestamp, IDs, summary, and data offsets/sizes
- Metadata DB (SQLite):
  - WAL cursor
  - Fresh/baked block rows
  - record counts, time ranges, bloom filters
- Compaction worker:
  - Converts sealed fresh blocks into baked blocks
  - Updates metadata with baked block stats and bloom filters
- Query coordinator:
  - Uses metadata time range + bloom filter pruning
  - Dispatches block groups to a pre-started worker thread pool
  - Workers load baked headers only first
  - Similarity scoring over summaries via FAISS
  - Applies similarity/time/agent/session filtering before data-section reads
  - Returns `COMPLETE`, `INCOMPLETE`, or `INTERNAL_ERROR`

## Build

```bash
cmake -S . -B build
cmake --build build -j8
```

FAISS is a required dependency.

## Dependencies

Kalki requires:

- C++20 compiler (`clang++` or `g++`)
- CMake (>= 3.23)
- Abseil
- Protobuf (+ `protoc`)
- gRPC
- SQLite3
- libcurl
- zstd (optional compression, enabled if found)
- FAISS (required)
- GoogleTest (for tests)

### macOS (Homebrew)

```bash
brew update
brew install cmake pkg-config abseil protobuf grpc sqlite zstd curl faiss googletest
```

If CMake cannot find FAISS config, set:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="$(brew --prefix)"
```

### Linux

#### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config \
  libabsl-dev protobuf-compiler libprotobuf-dev \
  libgrpc++-dev grpc-proto \
  libsqlite3-dev libcurl4-openssl-dev libzstd-dev \
  libgtest-dev
```

Install FAISS:

```bash
sudo apt-get install -y libfaiss-dev
```

If your distro release does not provide `libfaiss-dev`, build/install FAISS from source and pass
`-DCMAKE_PREFIX_PATH` (or `-DFaiss_DIR=...`) to CMake.

#### Fedora/RHEL (dnf)

```bash
sudo dnf install -y \
  gcc-c++ cmake pkgconf-pkg-config \
  abseil-cpp-devel protobuf-devel protobuf-compiler \
  grpc-devel sqlite-devel libcurl-devel libzstd-devel \
  gtest-devel
```

Install FAISS:

```bash
sudo dnf install -y faiss-devel
```

If `faiss-devel` is unavailable, install FAISS from source and point CMake at it with
`-DCMAKE_PREFIX_PATH` or `-DFaiss_DIR`.

## Run

```bash
./build/src/server/kalkid \
  --data_dir=./data \
  --wal_path=./data/wal.log \
  --metadata_db_path=./data/metadata.db \
  --fresh_block_dir=./data/blocks/fresh \
  --baked_block_dir=./data/blocks/baked \
  --grpc_listen_address=0.0.0.0:8080 \
  --llm_api_key="YOUR_API_KEY" \
  --llm_model="gemini-1.5-flash"
```

## Project layout

- `proto/`: API and internal record schemas
- `include/kalki/`: package interfaces
- `src/common`: config, logging, thread-pool
- `src/storage`: WAL, bloom filters, fresh/baked IO, compression
- `src/metadata`: SQLite metadata store
- `src/llm`: LLM abstraction + Gemini implementation
- `src/workers`: ingestion and compaction workers
- `src/query`: similarity + coordinator
- `src/core`: `DatabaseEngine` orchestration
- `src/api`: gRPC service implementation
- `src/server`: binary entrypoint

## Notes

- Structured logs are emitted with `component=<...> event=<...>` fields in both INFO and DEBUG (`DLOG`) paths.
