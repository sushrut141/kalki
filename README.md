# Kalki: The Ultra-High Throughput Semantic WAL for Agents

**Kalki** is a purpose-built database for the high-concurrency era of autonomous agents.
Kalki provides single digit ms latencies for store retrieve operations for agent conversation
logs at scale. It's built with agent orchestrators in mind that generate copius amount of data
that other agents need to store and query quickly.

Kalki takes an opinionated approach to agent logs.
It expects an agent_id representing the agent storing the conversation, session_id representing
the current action the agent is undertaking and the raw text of the agent's output.
Records within Kalki can be queried using natural language along with addition filters on 
agent, session ID and timestamps.


## 🚀 Motivation: The Context Crisis

Autonomous agents running 24/7 produce millions of log lines. 
Existing solutions (Standard RAG, Vector DBs, or Local Markdown files) force a trade-off:
 - The Noise Problem: Indexing every raw "thought" token makes vector search imprecise.
 - The Decompression Tax: To retrieve one specific context, current databases often have to decompress entire data pages, killing throughput.
 - The Token Tax: Agents shouldn't have to read their entire history to find a single past decision.

Kalki solves this by treating agent history like a Log-Structured Merge-Tree (LSM),
where the index is embedding-driven and the raw data is retrieved only on demand.

## ✨ Key Features

1. Tablet-Based Storage Architecture  
Data is persisted in tablets with two sections:
- Header: metadata (`agent_id`, `session_id`, time range), vector embeddings, and offsets.
- Body: compressed raw conversation payloads.

2. Semantic Predicate Pushdown  
The query engine first filters blocks using metadata (time + bloom filters), then runs FAISS similarity over stored embeddings.

3. O(1) Payload Retrieval  
After finding matches in headers, Kalki uses byte offsets to fetch only matching raw payloads from block bodies.

4. Background Compaction
Writes are WAL append-first. Ingestion + compaction workers bake blocks asynchronously.

5. Flexible Retrieval Signal
If caller supplies a summary, Kalki embeds the summary for retrieval. Otherwise, Kalki embeds the raw payload.

## 🚀 Performance Metrics
Kalki is built in C++20 for raw hardware efficiency. These numbers represent raw throughput (StoreLog operations/second) on standard hardware.

| DB Operation | Throughput | Performance Status |
| :--- | :--- | :--- |
| **StoreLog (WAL Write)** | **2,800+ QPS** | **High-Throughput** |
| **CommitTablet (Compaction)** | <100ms | Background Processed |
| **Search (Header Scan)** | O(HeaderSize) | Ultra-Fast In-Memory |

You can run the benchmarks using the benchmark command in the scripts directory.  

## Build

```bash
cmake -S . -B build
cmake --build build -j8
```

FAISS is a required dependency.

## Run Tests

```bash
ctest --test-dir build --output-on-failure
```

## Dependencies

Kalki requires:

- C++20 compiler (`clang++` or `g++`)
- CMake (>= 3.23)
- Abseil
- Protobuf (+ `protoc`)
- gRPC
- SQLite3
- libcurl
- zstd (required; all records are compressed)
- FAISS (required)
- llama.cpp (required for local embedding inference)
- GoogleTest (for tests)
- local embedding model file (required for similarity queries)

### macOS (Homebrew)

```bash
brew update
brew install cmake pkg-config abseil protobuf grpc sqlite zstd curl faiss googletest llama.cpp
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

Install llama.cpp if your distro does not package it:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cmake -S llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=ON
cmake --build llama.cpp/build -j8
cmake --install llama.cpp/build --prefix /usr/local
```

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

Install llama.cpp from source if no distro package is available (same steps as Ubuntu/Debian above).

### Local Embedding Model

Kalki stores record embeddings once at ingestion/compaction time and reuses them for queries.
Only the incoming query is embedded at query time.

Recommended small local model:

- `all-minilm:v2` as GGUF (`all-minilm-v2.gguf`), loaded directly in-memory by Kalki
  through `llama.cpp`.

Setup:

```bash
mkdir -p third_party/models
curl -L \
  "https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q4_K_M.gguf" \
  -o third_party/models/all-minilm-v2.gguf
```

Default config in Kalki:

- `--embedding_model_path=./third_party/models/all-minilm-v2.gguf`
- `--embedding_threads=2`
- `--wal_trim_threshold_records=0` (uses `2 * max_records_per_fresh_block` when zero)

## Run

```bash
./build/src/server/kalkid \
  --data_dir=./data \
  --wal_path=./data/wal.log \
  --metadata_db_path=./data/metadata.db \
  --fresh_block_dir=./data/blocks/fresh \
  --baked_block_dir=./data/blocks/baked \
  --grpc_listen_address=0.0.0.0:8080 \
  --statusz_listen_address=0.0.0.0:8081
```

Status page:

```bash
curl -s http://127.0.0.1:8081/statusz
```

## Project layout

- `proto/`: API and internal record schemas
- `include/kalki/`: package interfaces
- `src/common`: config, logging, thread-pool
- `src/storage`: WAL, bloom filters, fresh/baked IO, compression
- `src/metadata`: SQLite metadata store
- `src/llm`: embedding clients
- `src/workers`: ingestion and compaction workers
- `src/query`: similarity + coordinator
- `src/core`: `DatabaseEngine` orchestration
- `src/api`: gRPC service implementation
- `src/server`: binary entrypoint

## Notes

- Structured logs are emitted with `component=<...> event=<...>` fields in both INFO and DEBUG (`DLOG`) paths.
