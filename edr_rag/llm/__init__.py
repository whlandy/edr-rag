"""生成侧 (P4 Phase D) —— LLM provider 抽象与自托管端点客户端。

P0–P3 的「RAG」没有 G：``synthesize_answer(llm=...)`` 是个可注入 callable，
但仓库里没有任何实现，默认永远走模板拼接。本包补齐这一层。

部署形态锁定为**内网离线 + 自托管 OpenAI 兼容端点** (vLLM / Ollama / TGI /
Xinference)，见 P4-HARDENING-DESIGN.md §3.1。

用法::

    from edr_rag.llm import get_llm_client, make_synthesize_callable

    client = get_llm_client()          # 未配置端点时返回 None
    if client is not None:
        llm = make_synthesize_callable(client)
        answer = synthesize_answer(request, context, llm=llm)

配置全部走环境变量，见 ``openai_compat`` 模块 docstring。未配置
``EDR_RAG_LLM_BASE_URL`` 时 ``get_llm_client()`` 返回 None，整条链路自动
退回 P2 的确定性组装路径 —— 这保证了「没接模型也能跑」。
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional

from .base import EchoLLMClient, LLMClient, LLMMessage, LLMResponse
from .openai_compat import OpenAICompatClient
from .prompt import (
    SYSTEM_PROMPT,
    build_messages,
    extract_json,
    parse_structured_answer,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "EchoLLMClient",
    "OpenAICompatClient",
    "SYSTEM_PROMPT",
    "build_messages",
    "extract_json",
    "parse_structured_answer",
    "get_llm_client",
    "build_llm_client",
    "reset_llm_client",
    "make_synthesize_callable",
    "llm_configured",
]

_client: Optional[LLMClient] = None
_client_checked = False
_client_lock = threading.Lock()


def llm_configured() -> bool:
    """是否配置了生成端点。未配置即表示运维选择只用组装路径。"""
    return bool(os.environ.get("EDR_RAG_LLM_BASE_URL", "").strip())


def build_llm_client() -> Optional[LLMClient]:
    """按环境变量构造一个新的 LLM 客户端；未配置或配置非法时返回 None。"""
    base_url = os.environ.get("EDR_RAG_LLM_BASE_URL", "").strip()
    if not base_url:
        return None
    model = os.environ.get("EDR_RAG_LLM_MODEL", "").strip()
    if not model:
        logger.warning(
            "EDR_RAG_LLM_BASE_URL is set but EDR_RAG_LLM_MODEL is empty; "
            "generation stays on the assembly path"
        )
        return None
    try:
        return OpenAICompatClient(
            base_url=base_url,
            model=model,
            api_key=os.environ.get("EDR_RAG_LLM_API_KEY") or None,
            timeout=float(os.environ.get("EDR_RAG_LLM_TIMEOUT", "30")),
            retries=int(os.environ.get("EDR_RAG_LLM_RETRIES", "1")),
            max_tokens=int(os.environ.get("EDR_RAG_LLM_MAX_TOKENS", "1024")),
        )
    except (ValueError, TypeError) as exc:
        logger.warning("invalid LLM configuration (%s); using assembly path", exc)
        return None


def get_llm_client() -> Optional[LLMClient]:
    """返回进程内共享的 LLM 客户端 (惰性构造)；未配置返回 None。"""
    global _client, _client_checked
    if _client_checked:
        return _client
    with _client_lock:
        if _client_checked:
            return _client
        _client = build_llm_client()
        _client_checked = True
        if _client is not None:
            logger.info(
                "LLM generation enabled: %s (model=%s)",
                getattr(_client, "base_url", "?"), getattr(_client, "model", "?"),
            )
    return _client


def reset_llm_client() -> None:
    """重置共享客户端 (测试 / 改配置后调用)。"""
    global _client, _client_checked
    with _client_lock:
        if _client is not None:
            try:
                _client.close()
            except Exception:  # pragma: no cover - best-effort
                pass
        _client = None
        _client_checked = False


def make_synthesize_callable(
    client: LLMClient,
    max_tokens: int = 1024,
) -> Callable[[str, str], str]:
    """把 LLMClient 适配成 ``synthesize_answer`` 期望的 ``(请求, 上下文) -> str``。

    保留 P2-1 的注入契约不变（向后兼容既有调用方与测试），同时让真实模型
    可以接进去。上下文文本已由 ``synthesize_answer`` 用
    ``render_isolated_context`` 清洗并加过围栏，这里直接作为 user 消息。

    生成失败时返回空串 —— ``synthesize_answer`` 见空即降级到组装路径。
    """

    def _call(user_request: str, context_text: str) -> str:
        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=context_text),
        ]
        resp = client.complete(messages, max_tokens=max_tokens)
        if not resp.ok:
            return ""
        if resp.truncated:
            # 截断的操作步骤是危险的：宁可退回确定性组装，也不给半条流程。
            logger.warning(
                "LLM completion truncated (finish_reason=length); "
                "discarding and falling back to assembly"
            )
            return ""
        parsed = parse_structured_answer(resp.text)
        if parsed is None:
            # 模型没给 JSON —— 当作纯文本 answer 使用（仍比组装路径自然）。
            return resp.text
        if not parsed.get("evidence_sufficient", True):
            # 模型自己判定证据不足：交回组装路径，由 P2-3 的确定性拒答逻辑
            # 统一表达，避免两套拒答口径。
            return ""
        return parsed.get("answer") or resp.text

    return _call
