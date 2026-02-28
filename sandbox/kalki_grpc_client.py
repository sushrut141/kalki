from __future__ import annotations

import pathlib
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import grpc
from google.protobuf.timestamp_pb2 import Timestamp

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROTO_FILE = REPO_ROOT / "proto" / "kalki.proto"
GENERATED_DIR = pathlib.Path(__file__).resolve().parent / "generated"


def _ensure_generated() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    pb2 = GENERATED_DIR / "kalki_pb2.py"
    grpc_pb2 = GENERATED_DIR / "kalki_pb2_grpc.py"
    if pb2.exists() and grpc_pb2.exists():
        return

    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"-I{REPO_ROOT / 'proto'}",
        f"--python_out={GENERATED_DIR}",
        f"--grpc_python_out={GENERATED_DIR}",
        str(PROTO_FILE),
    ]
    subprocess.run(cmd, check=True)


def _import_stubs():
    _ensure_generated()
    if str(GENERATED_DIR) not in sys.path:
        sys.path.insert(0, str(GENERATED_DIR))

    import kalki_pb2  # type: ignore
    import kalki_pb2_grpc  # type: ignore

    return kalki_pb2, kalki_pb2_grpc


@dataclass
class QueryFilters:
    agent_id: Optional[str] = None
    session_id: Optional[str] = None


class KalkiGrpcClient:
    def __init__(self, target: str) -> None:
        self._kalki_pb2, self._kalki_pb2_grpc = _import_stubs()
        self._channel = grpc.insecure_channel(target)
        self._stub = self._kalki_pb2_grpc.AgentLogServiceStub(self._channel)

    def close(self) -> None:
        self._channel.close()

    def store_log(
        self, agent_id: str, session_id: str, conversation_log: str, summary: Optional[str] = None
    ):
        req = self._kalki_pb2.StoreLogRequest(
            agent_id=agent_id,
            session_id=session_id,
            conversation_log=conversation_log,
        )
        if summary:
            req.summary = summary
        ts = Timestamp()
        ts.GetCurrentTime()
        req.timestamp.CopyFrom(ts)
        return self._stub.StoreLog(req, timeout=5.0)

    def query_logs(self, caller_agent_id: str, query: str, filters: QueryFilters):
        req = self._kalki_pb2.QueryRequest(
            caller_agent_id=caller_agent_id,
            query=query,
        )
        if filters.agent_id:
            req.agent_id = filters.agent_id
        if filters.session_id:
            req.session_id = filters.session_id
        return self._stub.QueryLogs(req, timeout=10.0)
