# Kalki

Database designed for the high-concurrency needs of autonomous agents. 
Unlike traditional vector databases that index noisy, raw logs, Kalki uses a Tablet Architecture 
to decouple semantic indexing from raw data persistence.

## 🚀 Motivation: The Context Crisis

Autonomous agents running 24/7 produce millions of log lines. 
Existing solutions (Standard RAG, Vector DBs, or Local Markdown files) force a trade-off:
 - The Noise Problem: Indexing every raw "thought" token makes vector search imprecise.
 - The Decompression Tax: To retrieve one specific context, current databases often have to decompress entire data pages, killing throughput.
 - The Token Tax: Agents shouldn't have to read their entire history to find a single past decision.

Kalki solves this by treating agent history like a Log-Structured Merge-Tree (LSM), 
where the index is a high-signal summary and the raw data is retrieved only on demand.

## ✨ Key Features
1. Tablet-Based Storage ArchitectureData is persisted in "Tablets" consisting of two distinct sections:
 - The Header: Contains compressed 2-line summaries for every record, alongside metadata (Agent ID, Session ID, Time Range) and byte-offsets.
 - The Body: Contains the individual, heavily compressed raw conversation payloads.

2. Semantic Predicate Pushdown

The query engine filters tablets by structured metadata (e.g., session_id or timestamp) first. 
It then performs an in-memory FAISS vector search specifically on the decompressed 2-line summaries in the header.

3. $O(1)$ Random Access Retrieval

Once a semantic match is found in the header, it uses the stored offset to pluck the raw payload directly from the body. 
You only pay the decompression cost for the data you actually need.

4. Background Compaction

Write operations are $O(1)$ append-only logs. 
A background worker periodically chunks the logs and uses a small LLM to generate the 2-line summaries 
that populate the Tablet Headers, ensuring the "agent loop" is never blocked.

5. High Signal-to-Noise Retrieval

By embedding Summaries instead of Raw Logs, the vector space is "sharper." 
This leads to significantly higher recall and accuracy when agents are searching for past intents, decisions, or state changes.

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
- zstd (optional compression, enabled if found)
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

Kalki stores summary embeddings once at ingestion/compaction time and reuses them for queries.
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
