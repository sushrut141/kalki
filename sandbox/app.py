from __future__ import annotations

import html
import os
from typing import List

import grpc
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from sandbox.kalki_grpc_client import KalkiGrpcClient, QueryFilters

KALKI_GRPC_TARGET = os.getenv("KALKI_GRPC_TARGET", "127.0.0.1:8080")

app = FastAPI(title="Kalki Sandbox")


def _render_page(message: str = "", rows: List[str] | None = None) -> str:
    row_html = "".join(rows or [])
    return f"""
<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Kalki Sandbox</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 14px; }}
    form {{ border: 1px solid #ddd; padding: 10px; margin-bottom: 12px; }}
    input, textarea {{ width: 100%; margin-top: 6px; margin-bottom: 8px; box-sizing: border-box; }}
    button {{ padding: 5px 10px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; font-size: 13px; }}
    .msg {{ margin: 8px 0; color: #222; }}
  </style>
</head>
<body>
  <h2>Kalki Sandbox</h2>
  <div class=\"msg\">{html.escape(message)}</div>

  <form method=\"post\" action=\"/store\">
    <h3>Store Log</h3>
    <label>Agent ID</label>
    <input name=\"agent_id\" value=\"agent-ui\" required />
    <label>Session ID</label>
    <input name=\"session_id\" value=\"session-ui\" required />
    <label>Conversation Log</label>
    <textarea name=\"conversation_log\" rows=\"6\" required>Investigating a flaky build in CI.</textarea>
    <label>Summary For This Row (optional, used for indexing)</label>
    <textarea name=\"summary\" rows=\"3\" placeholder=\"Short summary from caller LLM\"></textarea>
    <button type=\"submit\">Store</button>
  </form>

  <form method=\"post\" action=\"/query\">
    <h3>Query Logs</h3>
    <label>Caller Agent ID</label>
    <input name=\"caller_agent_id\" value=\"sandbox-user\" required />
    <label>Natural Language Query</label>
    <input name=\"query\" value=\"What happened in the build investigation?\" required />
    <label>Filter Agent ID (optional)</label>
    <input name=\"agent_id\" />
    <label>Filter Session ID (optional)</label>
    <input name=\"session_id\" />
    <button type=\"submit\">Query</button>
  </form>

  <h3>Query Results</h3>
  <table>
    <thead><tr><th>Agent ID</th><th>Session ID</th><th>Timestamp</th><th>Raw Conversation Log</th></tr></thead>
    <tbody>
      {row_html if row_html else '<tr><td colspan="4">No results yet</td></tr>'}
    </tbody>
  </table>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _render_page("Connected to Kalki at " + KALKI_GRPC_TARGET)


@app.post("/store", response_class=HTMLResponse)
def store(
    agent_id: str = Form(...),
    session_id: str = Form(...),
    conversation_log: str = Form(...),
    summary: str = Form(""),
) -> str:
    client = KalkiGrpcClient(KALKI_GRPC_TARGET)
    try:
        resp = client.store_log(
            agent_id=agent_id,
            session_id=session_id,
            conversation_log=conversation_log,
            summary=summary.strip() or None,
        )
        return _render_page(f"StoreLog status={resp.status} error='{resp.error_message}'")
    except grpc.RpcError as exc:
        return _render_page(f"StoreLog RPC error: {exc.code()} {exc.details()}")
    finally:
        client.close()


@app.post("/query", response_class=HTMLResponse)
def query(
    caller_agent_id: str = Form(...),
    query: str = Form(...),
    agent_id: str = Form(""),
    session_id: str = Form(""),
) -> str:
    client = KalkiGrpcClient(KALKI_GRPC_TARGET)
    try:
        resp = client.query_logs(
            caller_agent_id=caller_agent_id,
            query=query,
            filters=QueryFilters(
                agent_id=agent_id.strip() or None,
                session_id=session_id.strip() or None,
            ),
        )
        rows = []
        for record in resp.records:
            seconds = record.timestamp.seconds
            nanos = record.timestamp.nanos
            ts = f"{seconds}.{nanos:09d}"
            rows.append(
                "<tr>"
                f"<td>{html.escape(record.agent_id)}</td>"
                f"<td>{html.escape(record.session_id)}</td>"
                f"<td>{html.escape(ts)}</td>"
                f"<td>{html.escape(record.raw_conversation_log)}</td>"
                "</tr>"
            )
        return _render_page(
            f"QueryLogs status={resp.status} records={len(resp.records)} error='{resp.error_message}'",
            rows,
        )
    except grpc.RpcError as exc:
        return _render_page(f"QueryLogs RPC error: {exc.code()} {exc.details()}")
    finally:
        client.close()
