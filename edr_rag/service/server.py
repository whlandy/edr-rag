"""edr-rag HTTP 服务 (P4 Phase G)。

P0–P3 只有 stdio MCP 与 CLI —— 没有 HTTP 接口、没有鉴权、没有限流、没有
健康检查，也就没法被内网里的别的系统（SOC 工作台、工单系统、值班机器人）
调用。这个模块补上这一层。

为什么用 stdlib ``ThreadingHTTPServer`` 而不是 FastAPI + uvicorn:
  部署形态是**内网离线**。fastapi + uvicorn + pydantic + starlette 一串
  依赖在客户现场意味着一串要审计、要随 CVE 升级的 wheel。这个服务的接口
  面很小（5 个只读端点 + 1 个反馈写入），QPS 也不高，stdlib 完全够用。
  接口层与业务层是分开的 —— 真需要换 uvicorn 时只改这一个文件。

端点：
  GET  /healthz            存活探针（不鉴权）
  GET  /readyz             就绪探针（不鉴权）—— 检查语料是否可加载
  POST /v1/search          检索
  POST /v1/ask             结构化答案（AskAnswer）
  GET  /v1/actions/{id}    按 action_id 精确获取
  POST /v1/feedback        执行反馈写入

鉴权：所有 /v1/* 端点要求 API key。**没有配置任何 key 时服务拒绝启动** ——
一个忘了配 key 的内网服务等于把整本手册对全网开放，静默放行是更糟的默认。
"""

from __future__ import annotations

import argparse
import json
import logging
import socketserver
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from .auth import Authenticator

logger = logging.getLogger(__name__)

#: 单次请求体上限，防止一个超大 body 把内存打满。
MAX_BODY_BYTES = 1 * 1024 * 1024

#: 超限时最多排空多少字节（好让客户端收到干净的 413 而不是 broken pipe）。
DRAIN_CAP = 8 * 1024 * 1024

#: 检索 top_k 的硬上限 —— 调用方传 top_k=100000 不该让服务卡死。
MAX_TOP_K = 100


class ServiceError(Exception):
    """带 HTTP 状态码的业务错误。"""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class EdrRagService:
    """把 RetrievalService / EdrRagAPI 包装成 HTTP 可调用的业务层。

    与传输层解耦：这里不认识 HTTP，只接受/返回 dict，便于单测与将来换框架。
    """

    def __init__(self, workdir: str | Path):
        from ..p2.api import EdrRagAPI
        from ..retrieval import RetrievalService

        self.workdir = Path(workdir)
        self.retrieval = RetrievalService(self.workdir)
        self.api = EdrRagAPI(self.workdir)
        self.started_at = time.time()

    # ---------- 探针 ----------

    def healthz(self) -> dict:
        return {"ok": True, "uptime_s": round(time.time() - self.started_at, 1)}

    def readyz(self) -> dict:
        """就绪 = 至少有一个语料文件可加载。

        存活与就绪分开：进程活着但 workdir 里没有产物时，不该被负载均衡
        送流量 —— 那只会给用户一串空结果。
        """
        corpora = {}
        ready = False
        for name in ("actions", "procedures", "sections", "chunks"):
            try:
                corpus = self.retrieval._corpus(name)
            except Exception:  # pragma: no cover - 防御性
                corpus = None
            count = len(corpus.records) if corpus is not None else 0
            corpora[name] = count
            if count:
                ready = True
        return {"ok": ready, "corpora": corpora, "workdir": str(self.workdir)}

    # ---------- 业务 ----------

    def search(self, payload: dict, acl=None) -> dict:
        query = _require_str(payload, "query")
        corpus_type = str(payload.get("corpus_type") or "actions")
        strategy = str(payload.get("strategy") or "keyword")
        top_k = _clamp_int(payload.get("top_k", 5), 1, MAX_TOP_K)
        try:
            results = self.retrieval.search(
                query,
                corpus_type=corpus_type,
                strategy=strategy,
                top_k=top_k,
                filters=payload.get("filters") or {},
                acl=acl,
            )
        except ValueError as exc:
            # 未知 corpus_type / strategy 是调用方错误，不是服务故障。
            raise ServiceError(400, str(exc)) from exc
        return {"query": query, "corpus_type": corpus_type,
                "strategy": strategy, "count": len(results), "results": results}

    def ask(self, payload: dict, acl=None) -> dict:
        query = _require_str(payload, "query")
        limit = _clamp_int(payload.get("limit", 5), 1, MAX_TOP_K)
        try:
            return self.api.ask_answer(
                query,
                limit=limit,
                strategy=str(payload.get("strategy") or "keyword"),
                min_confidence=payload.get("min_confidence"),
                confirm=payload.get("confirm"),
                acl=acl,
            )
        except ValueError as exc:
            raise ServiceError(400, str(exc)) from exc

    def get_action(self, action_id: str, acl=None) -> dict:
        results = self.retrieval.search(
            "", corpus_type="actions", filters={"id": action_id}, acl=acl
        )
        if not results:
            # 越权与不存在返回同一个 404 —— 用状态码区分等于确认了资源存在。
            raise ServiceError(404, f"action not found: {action_id}")
        return results[0]

    def feedback(self, payload: dict) -> dict:
        from ..p3.server import EdrRagAgent

        if not isinstance(payload.get("manual_id"), str):
            raise ServiceError(400, "manual_id is required")
        agent = EdrRagAgent(str(self.workdir))
        return {"feedback_id": agent.capture_feedback(payload), "status": "captured"}


# ---------- 参数校验 ----------


def _require_str(payload: dict, key: str) -> str:
    val = payload.get(key)
    if not isinstance(val, str) or not val.strip():
        raise ServiceError(400, f"'{key}' is required and must be a non-empty string")
    return val.strip()


def _clamp_int(value: Any, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ServiceError(400, f"expected an integer, got {value!r}") from None
    return max(lo, min(n, hi))


# ---------- HTTP 传输层 ----------


class _Handler(BaseHTTPRequestHandler):
    """请求分发。``service`` / ``auth`` 由 make_server 注入到类属性。"""

    service: EdrRagService = None       # type: ignore[assignment]
    auth: Authenticator = None          # type: ignore[assignment]
    server_version = "edr-rag/0.1"
    sys_version = ""                    # 不泄露 Python 版本

    # ----- 工具 -----

    def _send_json(self, status: int, payload: dict, extra_headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = self.headers.get("Content-Length")
        if length is None:
            return {}
        try:
            n = int(length)
        except ValueError:
            raise ServiceError(400, "invalid Content-Length") from None
        if n > MAX_BODY_BYTES:
            # 先把 body 排空再回 413。直接关连接会让客户端拿到 broken pipe
            # 而不是我们精心准备的错误信息——排查时那是一条毫无用处的线索。
            # 超过 DRAIN_CAP 的就不奉陪了，那种量级只可能是攻击或 bug。
            remaining = min(n, DRAIN_CAP)
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    break
                remaining -= len(chunk)
            raise ServiceError(413, "request body too large")
        raw = self.rfile.read(n)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ServiceError(400, f"malformed JSON body: {exc}") from exc
        if not isinstance(data, dict):
            raise ServiceError(400, "request body must be a JSON object")
        return data

    def _authorise(self):
        """鉴权 + 限流；失败时抛 ServiceError。"""
        principal = self.auth.authenticate(self.headers)
        if principal is None:
            raise ServiceError(401, "missing or invalid API key")
        ok, retry_after = self.auth.check_rate(principal)
        if not ok:
            err = ServiceError(429, "rate limit exceeded")
            err.retry_after = round(retry_after, 3)  # type: ignore[attr-defined]
            raise err
        return principal

    # ----- 分发 -----

    def do_GET(self):  # noqa: N802
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802
        self._dispatch("POST")

    def _dispatch(self, method: str):
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            # 探针不鉴权 —— k8s / 负载均衡拿不到 key。它们也不返回任何手册内容。
            if method == "GET" and path == "/healthz":
                return self._send_json(200, self.service.healthz())
            if method == "GET" and path == "/readyz":
                payload = self.service.readyz()
                return self._send_json(200 if payload["ok"] else 503, payload)

            principal = self._authorise()
            acl = principal.acl

            if method == "POST" and path == "/v1/search":
                return self._send_json(200, self.service.search(self._read_json(), acl))
            if method == "POST" and path == "/v1/ask":
                return self._send_json(200, self.service.ask(self._read_json(), acl))
            if method == "POST" and path == "/v1/feedback":
                return self._send_json(200, self.service.feedback(self._read_json()))
            if method == "GET" and path.startswith("/v1/actions/"):
                action_id = path[len("/v1/actions/"):]
                if not action_id:
                    raise ServiceError(400, "action_id is required")
                return self._send_json(200, self.service.get_action(action_id, acl))

            raise ServiceError(404, f"no route for {method} {path}")

        except ServiceError as exc:
            headers = {}
            if getattr(exc, "retry_after", None) is not None:
                headers["Retry-After"] = exc.retry_after  # type: ignore[attr-defined]
            self._send_json(exc.status, {"error": exc.message}, headers)
        except Exception as exc:  # noqa: BLE001
            # 内部异常不把栈泄露给调用方，但要完整记进服务端日志。
            logger.error("unhandled error on %s %s\n%s",
                         method, path, traceback.format_exc())
            self._send_json(500, {"error": "internal error",
                                  "type": type(exc).__name__})

    def log_message(self, fmt, *args):
        """走标准 logging，而不是往 stderr 糊。"""
        logger.info("%s - %s", self.address_string(), fmt % args)


class _Server(ThreadingHTTPServer):
    """跳过 HTTPServer.server_bind 里的 getfqdn() 反查。

    在受限/离线网络上那次反向解析要等 DNS 超时，实测能让启动卡几十秒 ——
    而离线正是本服务的目标部署环境。
    """

    daemon_threads = True
    allow_reuse_address = True

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


def make_server(
    workdir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8080,
    authenticator: Optional[Authenticator] = None,
) -> _Server:
    """构造 HTTP 服务实例（不启动）。

    Raises:
        RuntimeError: 未配置任何 API key。宁可起不来，也不要静默裸奔。
    """
    auth = authenticator or Authenticator()
    if not auth.enabled:
        raise RuntimeError(
            "no API keys configured — refusing to start.\n"
            "Set EDR_RAG_API_KEY, or EDR_RAG_API_KEYS_FILE pointing at a key file.\n"
            "(An unauthenticated manual service exposes the whole product manual "
            "to anyone who can reach the port.)"
        )

    handler = type("_BoundHandler", (_Handler,), {
        "service": EdrRagService(workdir),
        "auth": auth,
    })
    return _Server((host, int(port)), handler)


def serve(workdir: str | Path, host: str = "127.0.0.1", port: int = 8080) -> None:
    """启动并阻塞。"""
    server = make_server(workdir, host, port)
    logger.info("edr-rag HTTP service listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - 交互式退出
        logger.info("shutting down")
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="edr-rag HTTP service")
    parser.add_argument("--workdir", required=True, help="含 P0–P3 产物的工作目录")
    parser.add_argument("--host", default="127.0.0.1",
                        help="监听地址（默认只监听回环，需要对外时显式指定）")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        serve(args.workdir, args.host, args.port)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":  # pragma: no cover
    main()
