"""P4 Phase G —— HTTP 服务的鉴权、限流、路由与错误语义。

对应成熟度评估里的严重问题 10：只有 stdio MCP + CLI，无 HTTP API、无鉴权、
无限流、无并发模型、无健康检查。

测试对着**真实监听的 socket** 发请求，不 mock 传输层 —— 鉴权和限流这类
东西 mock 掉就等于没测。
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from edr_rag.security import ACLPolicy
from edr_rag.service import ApiPrincipal, Authenticator, TokenBucket, make_server

KEY_ADMIN = "test-key-admin"
KEY_TENANT_A = "test-key-tenant-a"
KEY_SLOW = "test-key-slow"


@pytest.fixture
def workdir(tmp_path):
    rows = [
        {"action_id": "network.configure_proxy", "id": "network.configure_proxy",
         "title": "配置代理服务器", "risk": "low", "status": "ready",
         "confidence": 0.9, "tenant": "tenant-a",
         "procedure": [{"order": 1, "action": "click", "target": "设置",
                        "source_instruction": "打开设置"}]},
        {"action_id": "system.reset", "id": "system.reset",
         "title": "重置系统", "risk": "high", "status": "ready",
         "confidence": 0.9, "tenant": "tenant-b",
         "procedure": [{"order": 1, "action": "click", "target": "重置",
                        "source_instruction": "点击重置"}]},
    ]
    (tmp_path / "actions.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def authenticator():
    return Authenticator({
        KEY_ADMIN: ApiPrincipal(key_id="admin"),
        KEY_TENANT_A: ApiPrincipal(
            key_id="tenant-a", acl=ACLPolicy(tenants={"tenant-a"})
        ),
        KEY_SLOW: ApiPrincipal(key_id="slow", rate_limit=1.0, burst=1),
    })


@pytest.fixture
def base_url(workdir, authenticator):
    server = make_server(workdir, host="127.0.0.1", port=0,
                         authenticator=authenticator)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.02), daemon=True
    )
    thread.start()
    host, port = server.server_address[0], server.server_address[1]
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _request(url, method="GET", body=None, key=None, timeout=5):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8")), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"raw": raw}
        return exc.code, payload, dict(exc.headers)


# ---------- 启动前置条件 ----------


def test_refuses_to_start_without_api_keys(workdir, monkeypatch):
    """AC-11 的前提：没配 key 就不许起来，而不是静默裸奔。"""
    monkeypatch.delenv("EDR_RAG_API_KEY", raising=False)
    monkeypatch.delenv("EDR_RAG_API_KEYS_FILE", raising=False)
    with pytest.raises(RuntimeError, match="no API keys configured"):
        make_server(workdir, port=0, authenticator=Authenticator({}))


# ---------- 探针 ----------


def test_healthz_needs_no_auth(base_url):
    status, payload, _ = _request(f"{base_url}/healthz")
    assert status == 200 and payload["ok"] is True


def test_readyz_reports_corpora(base_url):
    status, payload, _ = _request(f"{base_url}/readyz")
    assert status == 200
    assert payload["ok"] is True
    assert payload["corpora"]["actions"] == 2


def test_readyz_503_when_no_corpus(tmp_path, authenticator):
    """进程活着但没有产物时不该被送流量。"""
    server = make_server(tmp_path, host="127.0.0.1", port=0,
                         authenticator=authenticator)
    t = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.02),
                         daemon=True)
    t.start()
    try:
        host, port = server.server_address[0], server.server_address[1]
        status, payload, _ = _request(f"http://{host}:{port}/readyz")
        assert status == 503 and payload["ok"] is False
    finally:
        server.shutdown()
        server.server_close()
        t.join(timeout=2)


# ---------- 鉴权 (AC-11) ----------


def test_missing_key_is_401(base_url):
    status, payload, _ = _request(f"{base_url}/v1/search", "POST", {"query": "代理"})
    assert status == 401
    assert "API key" in payload["error"]


def test_wrong_key_is_401(base_url):
    status, _, _ = _request(f"{base_url}/v1/search", "POST",
                            {"query": "代理"}, key="totally-wrong")
    assert status == 401


def test_valid_key_succeeds(base_url):
    status, payload, _ = _request(f"{base_url}/v1/search", "POST",
                                  {"query": "代理"}, key=KEY_ADMIN)
    assert status == 200
    assert payload["count"] >= 1


def test_x_api_key_header_also_works(base_url):
    req = urllib.request.Request(
        f"{base_url}/v1/search",
        data=json.dumps({"query": "代理"}).encode(),
        headers={"Content-Type": "application/json", "X-API-Key": KEY_ADMIN},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200


# ---------- 限流 (AC-11) ----------


def test_rate_limit_returns_429(base_url):
    """burst=1, rate=1/s：连打必然触发 429。"""
    codes = [
        _request(f"{base_url}/v1/search", "POST", {"query": "代理"}, key=KEY_SLOW)[0]
        for _ in range(6)
    ]
    assert 200 in codes, "第一个请求应放行"
    assert 429 in codes, f"限流没有生效: {codes}"


def test_429_carries_retry_after(base_url):
    for _ in range(6):
        status, _, headers = _request(
            f"{base_url}/v1/search", "POST", {"query": "代理"}, key=KEY_SLOW
        )
        if status == 429:
            assert "Retry-After" in headers
            return
    pytest.fail("never hit the rate limit")


def test_unlimited_key_is_not_throttled(base_url):
    codes = [
        _request(f"{base_url}/v1/search", "POST", {"query": "代理"}, key=KEY_ADMIN)[0]
        for _ in range(10)
    ]
    assert all(c == 200 for c in codes)


# ---------- ACL 绑定到调用方身份 ----------


def test_key_acl_scopes_results(base_url):
    """key 决定能看到什么 —— 而不是调用方自己声明自己是谁。"""
    _, admin_payload, _ = _request(f"{base_url}/v1/search", "POST",
                                   {"query": "配置", "top_k": 10}, key=KEY_ADMIN)
    _, scoped_payload, _ = _request(f"{base_url}/v1/search", "POST",
                                    {"query": "配置", "top_k": 10}, key=KEY_TENANT_A)
    admin_ids = {r["id"] for r in admin_payload["results"]}
    scoped_ids = {r["id"] for r in scoped_payload["results"]}
    assert scoped_ids <= admin_ids
    assert "system.reset" not in scoped_ids


def test_get_action_404_hides_existence(base_url):
    """越权与不存在返回同一个 404 —— 用状态码区分等于确认资源存在。"""
    denied, _, _ = _request(f"{base_url}/v1/actions/system.reset", key=KEY_TENANT_A)
    missing, _, _ = _request(f"{base_url}/v1/actions/no.such.action", key=KEY_TENANT_A)
    assert denied == missing == 404


def test_get_action_allowed(base_url):
    status, payload, _ = _request(
        f"{base_url}/v1/actions/network.configure_proxy", key=KEY_TENANT_A
    )
    assert status == 200 and payload["id"] == "network.configure_proxy"


# ---------- 路由与错误语义 ----------


def test_ask_returns_structured_answer(base_url):
    status, payload, _ = _request(f"{base_url}/v1/ask", "POST",
                                  {"query": "如何配置代理"}, key=KEY_ADMIN)
    assert status == 200
    for key in ("answer", "steps", "citations", "answer_status", "confidence"):
        assert key in payload


def test_unknown_route_is_404(base_url):
    status, _, _ = _request(f"{base_url}/v1/nope", key=KEY_ADMIN)
    assert status == 404


def test_missing_query_is_400(base_url):
    status, payload, _ = _request(f"{base_url}/v1/search", "POST", {}, key=KEY_ADMIN)
    assert status == 400
    assert "query" in payload["error"]


def test_malformed_json_is_400(base_url):
    req = urllib.request.Request(
        f"{base_url}/v1/search", data=b"{not json",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {KEY_ADMIN}"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        pytest.fail("expected 400")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400


def test_unknown_corpus_type_is_400_not_500(base_url):
    """调用方传错参数是 4xx，不是服务故障。"""
    status, _, _ = _request(f"{base_url}/v1/search", "POST",
                            {"query": "x", "corpus_type": "bogus"}, key=KEY_ADMIN)
    assert status == 400


def test_top_k_is_clamped(base_url):
    """top_k=1000000 不该让服务卡死。"""
    status, payload, _ = _request(f"{base_url}/v1/search", "POST",
                                  {"query": "代理", "top_k": 10 ** 9}, key=KEY_ADMIN)
    assert status == 200
    assert len(payload["results"]) <= 100


def test_oversized_body_is_413(base_url):
    big = {"query": "x" * (2 * 1024 * 1024)}
    status, _, _ = _request(f"{base_url}/v1/search", "POST", big, key=KEY_ADMIN)
    assert status == 413


def test_server_does_not_leak_python_version(base_url):
    _, _, headers = _request(f"{base_url}/healthz")
    assert "Python" not in headers.get("Server", "")


# ---------- 并发安全 ----------


def test_concurrent_requests_are_safe(base_url):
    """多线程并发打同一个服务，不应出错或串数据。"""
    results: list[tuple[int, int]] = []
    lock = threading.Lock()

    def _worker():
        status, payload, _ = _request(
            f"{base_url}/v1/search", "POST", {"query": "代理"}, key=KEY_ADMIN
        )
        with lock:
            results.append((status, payload.get("count", -1)))

    threads = [threading.Thread(target=_worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    assert len(results) == 20
    assert all(s == 200 for s, _ in results), results
    # 同一个查询在并发下必须给出一致结果
    assert len({c for _, c in results}) == 1


# ---------- 令牌桶单元 ----------


def test_token_bucket_allows_burst_then_throttles():
    bucket = TokenBucket(rate=1.0, capacity=3)
    assert [bucket.allow()[0] for _ in range(3)] == [True, True, True]
    ok, retry_after = bucket.allow()
    assert ok is False and retry_after > 0


def test_token_bucket_unlimited_when_rate_zero():
    bucket = TokenBucket(rate=0.0, capacity=1)
    assert all(bucket.allow()[0] for _ in range(100))


def test_token_bucket_refills(monkeypatch):
    bucket = TokenBucket(rate=100.0, capacity=1)
    assert bucket.allow()[0] is True
    assert bucket.allow()[0] is False
    import time as _t

    _t.sleep(0.05)
    assert bucket.allow()[0] is True


# ---------- Authenticator 配置 ----------


def test_authenticator_from_env_single_key(monkeypatch):
    monkeypatch.setenv("EDR_RAG_API_KEY", "env-key")
    monkeypatch.delenv("EDR_RAG_API_KEYS_FILE", raising=False)
    auth = Authenticator()
    assert auth.enabled

    class _H(dict):
        def get(self, k, default=None):
            return dict.get(self, k, default)

    assert auth.authenticate(_H({"Authorization": "Bearer env-key"})) is not None
    assert auth.authenticate(_H({"Authorization": "Bearer nope"})) is None


def test_authenticator_from_key_file(tmp_path, monkeypatch):
    key_file = tmp_path / "keys.json"
    key_file.write_text(json.dumps({"keys": [
        {"key": "k1", "id": "soc", "tenants": ["t1"], "rate_limit": 5},
    ]}), encoding="utf-8")
    monkeypatch.setenv("EDR_RAG_API_KEYS_FILE", str(key_file))
    monkeypatch.delenv("EDR_RAG_API_KEY", raising=False)
    auth = Authenticator()

    class _H(dict):
        def get(self, k, default=None):
            return dict.get(self, k, default)

    principal = auth.authenticate(_H({"X-API-Key": "k1"}))
    assert principal is not None
    assert principal.key_id == "soc"
    assert principal.acl.tenants == {"t1"}


def test_authenticator_disabled_without_config(monkeypatch):
    monkeypatch.delenv("EDR_RAG_API_KEY", raising=False)
    monkeypatch.delenv("EDR_RAG_API_KEYS_FILE", raising=False)
    assert Authenticator().enabled is False
