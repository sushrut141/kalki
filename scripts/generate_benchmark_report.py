#!/usr/bin/env python3
import json
import pathlib
import sys


def _find_benchmark(results, name):
    for row in results.get("benchmarks", []):
        row_name = row.get("name", "")
        if row_name == name or row_name.startswith(name + "/"):
            return row
    return None


def _counter(row, key):
    if row is None:
        return 0.0
    value = row.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    counters = row.get("counters", {})
    if isinstance(counters, dict) and isinstance(counters.get(key), (int, float)):
        return float(counters.get(key))
    return 0.0


def main():
    if len(sys.argv) != 3:
        print("usage: generate_benchmark_report.py <input_json> <output_md>", file=sys.stderr)
        return 1

    input_path = pathlib.Path(sys.argv[1])
    output_path = pathlib.Path(sys.argv[2])

    data = json.loads(input_path.read_text())
    store = _find_benchmark(data, "BM_AgentLogService_StoreLog")
    query = _find_benchmark(data, "BM_AgentLogService_QueryLogs")

    content = "\n".join(
        [
            "# AgentLogService Benchmark Report",
            "",
            f"- StoreLog QPS: {_counter(store, 'qps'):.2f}",
            f"- StoreLog p50 latency (ms): {_counter(store, 'p50_ms'):.2f}",
            f"- StoreLog p90 latency (ms): {_counter(store, 'p90_ms'):.2f}",
            f"- QueryLogs QPS: {_counter(query, 'qps'):.2f}",
            f"- Query p50 latency (ms): {_counter(query, 'p50_ms'):.2f}",
            f"- Query p90 latency (ms): {_counter(query, 'p90_ms'):.2f}",
            "",
        ]
    )
    output_path.write_text(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
