#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SANDBOX_DIR="${ROOT_DIR}/sandbox"
VENV_DIR="${SANDBOX_DIR}/.venv"
DATA_DIR="${SANDBOX_DIR}/.sandbox_data"
LOG_DIR="${SANDBOX_DIR}/logs"
EMBEDDING_MODEL_PATH="${ROOT_DIR}/third_party/models/all-minilm-v2.gguf"

KALKI_GRPC_ADDR="${KALKI_GRPC_ADDR:-127.0.0.1:8080}"
KALKI_STATUSZ_ADDR="${KALKI_STATUSZ_ADDR:-127.0.0.1:8081}"
SANDBOX_HTTP_ADDR="${SANDBOX_HTTP_ADDR:-127.0.0.1:8000}"
KALKI_BUILD_TYPE="${KALKI_BUILD_TYPE:-Debug}"
KALKI_BUILD_DIR="${KALKI_BUILD_DIR:-${ROOT_DIR}/build-debug}"
SANDBOX_WAL_READ_BATCH_SIZE="${SANDBOX_WAL_READ_BATCH_SIZE:-1}"
SANDBOX_MAX_RECORDS_PER_FRESH_BLOCK="${SANDBOX_MAX_RECORDS_PER_FRESH_BLOCK:-100}"
SANDBOX_PUSHER_MAX_INGESTED_RECORDS="${SANDBOX_PUSHER_MAX_INGESTED_RECORDS:-500}"

if [[ ! -f "${EMBEDDING_MODEL_PATH}" ]]; then
  echo "error: embedding model not found at ${EMBEDDING_MODEL_PATH}"
  echo "expected model: third_party/models/all-minilm-v2.gguf"
  echo "download it first or fix the model path before running sandbox"
  exit 1
fi

# Reset sandbox runtime state on every startup.
rm -rf "${DATA_DIR}" "${LOG_DIR}"
mkdir -p "${DATA_DIR}" "${DATA_DIR}/blocks/fresh" "${DATA_DIR}/blocks/baked" "${LOG_DIR}"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
pip3 install --upgrade pip
if ! pip3 install --only-binary=:all: -r "${SANDBOX_DIR}/requirements.txt"; then
  echo "error: failed installing Python wheels for sandbox dependencies"
  echo "python: $(python3 --version 2>&1)"
  echo "hint: grpcio/grpcio-tools need wheel support for your interpreter"
  exit 1
fi

cmake -S "${ROOT_DIR}" -B "${KALKI_BUILD_DIR}" -DCMAKE_BUILD_TYPE="${KALKI_BUILD_TYPE}"
cmake --build "${KALKI_BUILD_DIR}" --target kalkid -j8

cleanup() {
  set +e
  if [[ -n "${PUSHER_PID:-}" ]]; then kill "${PUSHER_PID}" 2>/dev/null; fi
  if [[ -n "${FASTAPI_PID:-}" ]]; then kill "${FASTAPI_PID}" 2>/dev/null; fi
  if [[ -n "${KALKI_PID:-}" ]]; then kill "${KALKI_PID}" 2>/dev/null; fi
  wait "${PUSHER_PID:-}" 2>/dev/null
  wait "${FASTAPI_PID:-}" 2>/dev/null
  wait "${KALKI_PID:-}" 2>/dev/null
}
trap cleanup EXIT INT TERM

"${KALKI_BUILD_DIR}/src/server/kalkid" \
  --data_dir="${DATA_DIR}" \
  --wal_path="${DATA_DIR}/wal.log" \
  --metadata_db_path="${DATA_DIR}/metadata.db" \
  --fresh_block_dir="${DATA_DIR}/blocks/fresh" \
  --baked_block_dir="${DATA_DIR}/blocks/baked" \
  --grpc_listen_address="${KALKI_GRPC_ADDR}" \
  --statusz_listen_address="${KALKI_STATUSZ_ADDR}" \
  --embedding_model_path="${EMBEDDING_MODEL_PATH}" \
  --max_records_per_fresh_block="${SANDBOX_MAX_RECORDS_PER_FRESH_BLOCK}" \
  --wal_read_batch_size="${SANDBOX_WAL_READ_BATCH_SIZE}" \
  >"${LOG_DIR}/kalki.log" 2>&1 &
KALKI_PID=$!

sleep 1

KALKI_GRPC_TARGET="${KALKI_GRPC_ADDR}" \
  uvicorn --app-dir "${ROOT_DIR}" sandbox.app:app --host "${SANDBOX_HTTP_ADDR%:*}" --port "${SANDBOX_HTTP_ADDR##*:}" \
  >"${LOG_DIR}/fastapi.log" 2>&1 &
FASTAPI_PID=$!

python -m sandbox.push_rows --target "${KALKI_GRPC_ADDR}" --interval-ms 10 \
  --max-records "${SANDBOX_PUSHER_MAX_INGESTED_RECORDS}" \
  >"${LOG_DIR}/pusher.log" 2>&1 &
PUSHER_PID=$!

cat <<MSG
Kalki sandbox started.

- gRPC API:     ${KALKI_GRPC_ADDR}
- statusz:      http://${KALKI_STATUSZ_ADDR}/statusz
- UI:           http://${SANDBOX_HTTP_ADDR}/
- logs:         ${LOG_DIR}

Press Ctrl+C to stop all processes.
MSG

wait "${KALKI_PID}" "${FASTAPI_PID}" "${PUSHER_PID}"
