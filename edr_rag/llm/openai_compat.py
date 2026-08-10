"""自托管 OpenAI 兼容端点客户端 (P4 Phase D)。

对接 vLLM / Ollama / TGI / Xinference 等在内网自托管的推理服务 —— 它们
基本都暴露 ``POST {base_url}/chat/completions`` 的 OpenAI 兼容接口。

为什么不用 openai SDK:
  内网离线部署里，每多一个 wheel 就多一份版本适配与安全审计成本。这个
  接口的请求/响应结构很稳定，直接用 httpx（若可用）或 stdlib
  ``urllib.request`` 发一个 POST 就够了，没有必要为此拉进一整个 SDK。

配置 (环境变量)：
  EDR_RAG_LLM_BASE_URL   端点根地址，如 http://10.0.0.8:8000/v1  (必填，
                         未设置时 get_llm_client() 返回 None，链路走组装)
  EDR_RAG_LLM_MODEL      模型名，如 Qwen2.5-14B-Instruct
  EDR_RAG_LLM_API_KEY    可选；自托管服务常常不校验，留空即不发 Authorization
  EDR_RAG_LLM_TIMEOUT    单次请求超时秒数 (默认 30)
  EDR_RAG_LLM_RETRIES    失败重试次数 (默认 1；只重试超时/5xx/连接错误)
  EDR_RAG_LLM_MAX_TOKENS 生成上限 (默认 1024)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Optional

from .base import LLMClient, LLMMessage, LLMResponse

logger = logging.getLogger(__name__)

#: 会重试的 HTTP 状态码 —— 只重试服务端/瞬时问题，不重试 4xx 语义错误。
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _httpx():
    """返回 httpx 模块，缺失时 None (回落到 urllib)。"""
    try:
        import httpx  # type: ignore

        return httpx
    except Exception:  # pragma: no cover - optional dep
        return None


class OpenAICompatClient(LLMClient):
    """OpenAI 兼容 /chat/completions 客户端。

    契约：``complete`` **绝不抛异常**。连接失败、超时、非 2xx、响应畸形，
    一律返回 ``ok=False`` 的 LLMResponse，让调用方降级到组装路径。
    """

    name = "openai-compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        retries: int = 1,
        max_tokens: int = 1024,
    ):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or None
        self.timeout = float(timeout)
        self.retries = max(int(retries), 0)
        self.max_tokens = int(max_tokens)
        self._httpx_mod = _httpx()
        self._client = None

    # ---------- 请求构造 ----------

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _payload(
        self,
        messages: list[LLMMessage],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": int(max_tokens),
            # 手册操作步骤要可复现：默认 0 温度。
            "temperature": float(temperature),
            "stream": False,
        }

    # ---------- 调用 ----------

    def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if not messages:
            return LLMResponse.failure("no messages", model=self.model)
        payload = self._payload(
            messages,
            max_tokens if max_tokens is not None else self.max_tokens,
            temperature,
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        t0 = time.perf_counter()
        last_error = "unknown error"
        for attempt in range(self.retries + 1):
            ok, result, retryable = self._post(body)
            if ok:
                return self._parse(result, (time.perf_counter() - t0) * 1000.0)
            last_error = result
            if not retryable or attempt >= self.retries:
                break
            # 线性退避；内网服务通常恢复很快，不需要指数级等待。
            time.sleep(min(0.5 * (attempt + 1), 2.0))

        latency = (time.perf_counter() - t0) * 1000.0
        logger.warning("LLM call failed (%s); falling back to assembly", last_error)
        return LLMResponse.failure(last_error, model=self.model, latency_ms=latency)

    def _post(self, body: bytes) -> tuple[bool, Any, bool]:
        """发一次 POST。返回 (成功?, 响应dict或错误串, 是否可重试)。"""
        if self._httpx_mod is not None:
            return self._post_httpx(body)
        return self._post_urllib(body)

    def _post_httpx(self, body: bytes) -> tuple[bool, Any, bool]:
        httpx = self._httpx_mod
        try:
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout)
            resp = self._client.post(
                self.endpoint, content=body, headers=self._headers()
            )
            if resp.status_code >= 400:
                return (
                    False,
                    f"HTTP {resp.status_code}: {resp.text[:200]}",
                    resp.status_code in _RETRYABLE_STATUS,
                )
            return True, resp.json(), False
        except Exception as exc:  # noqa: BLE001 - 任何传输层问题都可重试
            return False, f"{type(exc).__name__}: {exc}", True

    def _post_urllib(self, body: bytes) -> tuple[bool, Any, bool]:
        req = urllib.request.Request(
            self.endpoint, data=body, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", "replace")
            return True, json.loads(raw), False
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:  # pragma: no cover - 防御性
                pass
            return False, f"HTTP {exc.code}: {detail}", exc.code in _RETRYABLE_STATUS
        except json.JSONDecodeError as exc:
            return False, f"malformed JSON response: {exc}", False
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}", True

    def _parse(self, data: Any, latency_ms: float) -> LLMResponse:
        """把 OpenAI 兼容响应解析成 LLMResponse (对畸形响应容错)。"""
        if not isinstance(data, dict):
            return LLMResponse.failure(
                "response is not a JSON object", model=self.model, latency_ms=latency_ms
            )
        choices = data.get("choices") or []
        if not choices or not isinstance(choices, list):
            return LLMResponse.failure(
                "response has no choices", model=self.model, latency_ms=latency_ms
            )
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") or {}
        text = ""
        if isinstance(message, dict):
            text = str(message.get("content") or "")
        if not text:
            # 少数服务把内容放在 text 字段 (legacy completion 形状)
            text = str(first.get("text") or "")
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        return LLMResponse(
            text=text.strip(),
            ok=bool(text.strip()),
            error=None if text.strip() else "empty completion",
            model=str(data.get("model") or self.model),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
            finish_reason=str(first.get("finish_reason") or ""),
            raw=data if len(str(data)) < 20000 else {},
        )

    def available(self) -> bool:
        """轻量探活：发一个 1 token 的请求，看端点是否应答。

        「可用」= 端点在网络上应答了。返回 4xx（比如模型名写错）也算可用 ——
        那是配置问题不是连通性问题，运维需要看到不同的报错。只有传输层失败
        （连不上 / 超时 / DNS）才算不可用。
        """
        probe = self.complete(
            [LLMMessage(role="user", content="ping")], max_tokens=1
        )
        if probe.ok:
            return True
        return str(probe.error or "").startswith("HTTP ")

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover
                pass
            self._client = None
