#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
OUT_DIR="${ROOT_DIR}/benchmark_results"
JSON_OUT="${OUT_DIR}/agent_log_service_benchmark.json"
MD_OUT="${OUT_DIR}/agent_log_service_benchmark.md"

mkdir -p "${OUT_DIR}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DKALKI_BUILD_BENCHMARKS=ON
cmake --build "${BUILD_DIR}" --target kalki_agent_log_service_benchmark -j8

"${BUILD_DIR}/benchmarks/kalki_agent_log_service_benchmark" \
  --benchmark_out="${JSON_OUT}" \
  --benchmark_out_format=json \
  --benchmark_report_aggregates_only=true

python3 "${ROOT_DIR}/scripts/generate_benchmark_report.py" "${JSON_OUT}" "${MD_OUT}"
echo "Benchmark report generated at ${MD_OUT}"
