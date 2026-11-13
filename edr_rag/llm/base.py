"""LLM provider 抽象 (P4 Phase D)。

背景：P0–P3 的「RAG」没有 G —— 全仓库没有任何 LLM 客户端，``synthesize_answer``
的 ``llm`` 参数只是一个可注入的 callable，从来没有任何实现传进去过，默认
永远走 ``_assemble_answer`` 模板拼接。严格说那是 retrieval + 模板渲染器。

本模块定义生成侧的最小契约。设计约束来自部署形态 (内网离线):

  - **零硬依赖**：HTTP 走 httpx（若可用）或 stdlib ``urllib.request``。
    不引入 openai / langchain 之类的 SDK —— 那些在离线环境里是额外的
    wheel 与版本地狱。
  - **绝不阻塞主链路**：端点不可达 / 超时 / 返回畸形，一律返回
    ``ok=False`` 的 LLMResponse，由调用方降级到确定性组装路径。
    生成侧故障不得让检索也不可用。
  - **可计量**：每次调用带 token 数与延迟，接入 P3-2 的 tracing。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMMessage:
    """一条对话消息。

    ``role`` 只允许 'system' / 'user' / 'assistant'。检索到的文档内容
    **必须**放进 user 角色的围栏里，绝不能进 system —— system 是指令区，
    让文档内容进入那里等于把指令权交给不可信数据 (P4 §3.3 边界规则)。
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """一次生成调用的结果。

    ``ok=False`` 时 ``text`` 为空、``error`` 说明原因；调用方据此降级。
    """

    text: str = ""
    ok: bool = True
    error: Optional[str] = None
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        """模型是否因为长度上限被截断 —— 截断的操作步骤是危险的。"""
        return self.finish_reason == "length"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error": self.error,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(self.latency_ms, 3),
            "finish_reason": self.finish_reason,
            "truncated": self.truncated,
        }

    @staticmethod
    def failure(error: str, model: str = "", latency_ms: float = 0.0) -> "LLMResponse":
        return LLMResponse(
            text="", ok=False, error=error, model=model, latency_ms=latency_ms
        )


class LLMInsufficientEvidenceError(Exception):
    """模型判定检索证据不足以回答用户请求。

    由 ``make_synthesize_callable`` 适配层在模型返回
    ``evidence_sufficient=false`` 时抛出，``synthesize_answer`` 捕获后把
    ``answer_status`` 置为 ``refused`` —— 与确定性拒答共用同一套状态口径，
    避免「模型说证据不足、组装路径却按 answered 输出编造步骤」的分裂。
    """


class LLMClient(ABC):
    """生成侧客户端的最小接口。

    实现方**必须**保证 ``complete`` 不抛异常 —— 所有失败都表达为
    ``LLMResponse.failure(...)``。这是主链路可用性的硬约束。
    """

    name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """把消息列表送给模型，返回生成结果 (绝不抛异常)。"""

    def available(self) -> bool:
        """端点是否可用 (默认乐观返回 True，实现方可覆盖做探活)。"""
        return True

    def close(self) -> None:  # noqa: B027 - 可选
        """释放连接资源。"""


class EchoLLMClient(LLMClient):
    """测试用的确定性假客户端 —— 回显最后一条 user 消息的前若干字符。

    存在的意义：让生成链路（prompt 构建、结构化解析、降级、计量）可以在
    没有任何模型服务的 CI 环境里被真实覆盖，而不是只能 mock 掉。
    """

    name = "echo"

    def __init__(self, prefix: str = "", fail: bool = False, max_chars: int = 400):
        self.prefix = prefix
        self.fail = fail
        self.max_chars = max_chars

    def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        if self.fail:
            return LLMResponse.failure("echo client configured to fail", model=self.name)
        last_user = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user = m.content
                break
        text = (self.prefix + last_user)[: self.max_chars]
        return LLMResponse(
            text=text,
            model=self.name,
            prompt_tokens=sum(len(m.content) for m in messages) // 4,
            completion_tokens=len(text) // 4,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            finish_reason="stop",
        )
