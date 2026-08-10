"""P4 Phase D —— 生成侧端到端集成测试（对着一个真的 HTTP 端点）。

单元测试能证明解析逻辑对，但证明不了「整条链路真的通」。这里起一个
stdlib 的 HTTP 服务模拟 vLLM / Ollama 的 OpenAI 兼容接口，让
``EdrRagAPI.ask_answer`` 真的发出 HTTP 请求、真的收到生成结果。

覆盖的正是内网离线部署最关心的三件事：
  1. 配好端点后，答案确实由模型生成（generated_by == 'llm'）。
  2. 端点挂掉时，检索仍然可用，答案自动退回确定性组装。
  3. 送给模型的请求里，检索文档不在 system 消息里（安全边界）。
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from edr_rag.llm import reset_llm_client
from edr_rag.p2.api import EdrRagAPI

#: 服务端记录收到的请求体，供测试断言 prompt 结构。
_RECEIVED: list[dict] = []


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    """最小 OpenAI 兼容端点。"""

    # 行为开关，由测试设置
    status = 200
    reply = "在「设置 → 网络」中配置代理服务器。"

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler 的接口
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            _RECEIVED.append(json.loads(raw))
        except json.JSONDecodeError:  # pragma: no cover
            _RECEIVED.append({"_unparsed": raw})

        if type(self).status != 200:
            self.send_response(type(self).status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "boom"}')
            return

        payload = {
            "model": "fake-qwen",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "answer": type(self).reply,
                        "steps": [
                            {"order": 1, "instruction": "打开设置", "target": "设置"},
                            {"order": 2, "instruction": "填入代理地址", "target": "代理"},
                        ],
                        "preconditions": ["管理员权限"],
                        "warnings": ["修改后需重启代理服务"],
                        "requires_confirmation": False,
                        "evidence_sufficient": True,
                    }, ensure_ascii=False),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 320, "completion_tokens": 88},
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # 静音测试输出
        pass


class _FastBindServer(ThreadingHTTPServer):
    """跳过 HTTPServer.server_bind 里的 socket.getfqdn() 反向解析。

    在受限网络（比如正是本项目目标场景的内网/离线机器）上，那次 FQDN 反查
    要等 DNS 超时 —— 实测让本文件的 fixture setup 花掉 35 秒。服务端名字对
    测试毫无意义，直接用绑定地址。
    """

    def server_bind(self):
        import socketserver

        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


@pytest.fixture
def fake_endpoint():
    """起一个真的 HTTP 服务，yield 其 base_url。"""
    _RECEIVED.clear()
    _FakeOpenAIHandler.status = 200
    server = _FastBindServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.02), daemon=True
    )
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def workdir(tmp_path):
    (tmp_path / "actions.jsonl").write_text(
        json.dumps({
            "action_id": "network.configure_proxy",
            "title": "配置代理服务器",
            "risk": "low",
            "status": "ready",
            "confidence": 0.9,
            "preconditions": ["管理员权限"],
            "procedure": [
                {"order": 1, "action": "click", "target": "设置",
                 "source_instruction": "打开设置页"},
            ],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture(autouse=True)
def _reset():
    reset_llm_client()
    yield
    reset_llm_client()


def _configure(monkeypatch, base_url):
    monkeypatch.setenv("EDR_RAG_LLM_BASE_URL", base_url)
    monkeypatch.setenv("EDR_RAG_LLM_MODEL", "fake-qwen")
    monkeypatch.setenv("EDR_RAG_LLM_TIMEOUT", "5")
    monkeypatch.setenv("EDR_RAG_LLM_RETRIES", "0")
    reset_llm_client()


def test_answer_is_generated_by_llm(fake_endpoint, workdir, monkeypatch):
    """配好端点后，答案确实来自模型而不是模板拼接。"""
    _configure(monkeypatch, fake_endpoint)
    result = EdrRagAPI(workdir).ask_answer("如何配置代理服务器")
    assert result["generated_by"] == "llm"
    assert "设置" in result["answer"]
    assert _RECEIVED, "端点没有收到请求"


def test_request_shape_is_openai_compatible(fake_endpoint, workdir, monkeypatch):
    _configure(monkeypatch, fake_endpoint)
    EdrRagAPI(workdir).ask_answer("如何配置代理服务器")
    req = _RECEIVED[-1]
    assert req["model"] == "fake-qwen"
    assert req["stream"] is False
    assert req["temperature"] == 0.0        # 操作步骤要可复现
    assert [m["role"] for m in req["messages"]] == ["system", "user"]


def test_retrieved_documents_stay_out_of_system_message(
    fake_endpoint, workdir, monkeypatch
):
    """安全边界的端到端验证：手册内容不得出现在 system 指令区。"""
    _configure(monkeypatch, fake_endpoint)
    EdrRagAPI(workdir).ask_answer("如何配置代理服务器")
    messages = {m["role"]: m["content"] for m in _RECEIVED[-1]["messages"]}
    assert "配置代理服务器" not in messages["system"]
    assert "配置代理服务器" in messages["user"]
    assert "不可信检索文档" in messages["user"]


def test_endpoint_failure_falls_back_to_assembly(
    fake_endpoint, workdir, monkeypatch
):
    """AC-8: 端点返回 5xx 时，检索仍可用，答案退回确定性组装。"""
    _configure(monkeypatch, fake_endpoint)
    _FakeOpenAIHandler.status = 500
    result = EdrRagAPI(workdir).ask_answer("如何配置代理服务器")
    assert result["generated_by"] == "assemble"
    # 关键：结构化字段仍然完整 —— 生成挂了不影响检索出来的步骤
    assert result["steps"]
    assert result["citations"]


def test_unreachable_endpoint_falls_back(workdir, monkeypatch):
    """端点完全连不上（内网服务没起）时同样降级，不抛异常。"""
    _configure(monkeypatch, "http://127.0.0.1:9/v1")
    result = EdrRagAPI(workdir).ask_answer("如何配置代理服务器")
    assert result["generated_by"] == "assemble"
    assert result["steps"]


def test_structured_fields_survive_llm_path(fake_endpoint, workdir, monkeypatch):
    """走 LLM 路径时，P2 的结构化字段（引用/风险/确认）不能丢。

    agent 消费的是这些字段，不是 answer 文本 —— 接了模型不能把它们弄没。
    """
    _configure(monkeypatch, fake_endpoint)
    result = EdrRagAPI(workdir).ask_answer("如何配置代理服务器")
    assert result["generated_by"] == "llm"
    for key in ("steps", "citations", "preconditions", "answer_status",
                "confidence", "requires_confirmation", "dangerous", "sources"):
        assert key in result, f"结构化字段 {key} 在 LLM 路径下丢失"
    assert result["citations"][0]["id"] == "network.configure_proxy"
