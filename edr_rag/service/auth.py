"""HTTP 服务的鉴权与限流 (P4 Phase G)。

纯 stdlib。设计约束同样来自内网离线部署：不引入 fastapi / starlette /
slowapi 之类的框架栈 —— 客户现场每多一个 wheel 就多一份审计与升级成本。

鉴权模型：
  静态 API key（``Authorization: Bearer <key>`` 或 ``X-API-Key``）。
  key 从环境变量或文件加载，可为每个 key 绑定 ACL 策略（租户/设备），
  于是「谁在问」直接决定「能检索到什么」——把 P3-3 的文档级 ACL 接到
  真实的调用方身份上，而不是让调用方自己声明自己是谁。

限流：
  每个 key 一个令牌桶。内网服务的目的不是防 DDoS，而是防止某个写错的
  循环把手册服务打挂、连累其它调用方。
"""

from __future__ import annotations

import hmac
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..security import ACLPolicy


@dataclass
class ApiPrincipal:
    """一个已鉴权的调用方。"""

    key_id: str
    acl: Optional[ACLPolicy] = None
    rate_limit: float = 0.0          # 每秒允许的请求数；<=0 表示不限
    burst: int = 0                   # 令牌桶容量；0 表示按 rate_limit 推导
    description: str = ""

    def __post_init__(self):
        if self.burst <= 0:
            self.burst = max(int(self.rate_limit * 2), 1)


class TokenBucket:
    """线程安全的令牌桶。"""

    def __init__(self, rate: float, capacity: int):
        self.rate = float(rate)
        self.capacity = max(int(capacity), 1)
        self._tokens = float(self.capacity)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def allow(self, cost: float = 1.0) -> tuple[bool, float]:
        """尝试取一个令牌。

        Returns:
            (是否放行, 建议的 Retry-After 秒数)。
        """
        if self.rate <= 0:
            return True, 0.0
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                self.capacity, self._tokens + (now - self._last) * self.rate
            )
            self._last = now
            if self._tokens >= cost:
                self._tokens -= cost
                return True, 0.0
            deficit = cost - self._tokens
            return False, max(deficit / self.rate, 0.001)


class Authenticator:
    """API key -> ApiPrincipal 的解析与限流。

    key 来源（优先级从高到低）：
      1. 构造参数 ``principals``
      2. ``EDR_RAG_API_KEYS_FILE`` 指向的 JSON 文件
      3. ``EDR_RAG_API_KEY`` 单个 key（最简部署，无 ACL、无限流）

    完全没有配置任何 key 时 ``enabled`` 为 False —— 此时服务**拒绝启动**
    而不是裸奔（见 server.py）。这比「默认放行」安全得多：一个忘了配 key
    的内网服务等于把整本手册对全网开放。

    key 文件格式::

        {
          "keys": [
            {"key": "...", "id": "soc-team", "tenants": ["t1"],
             "rate_limit": 10, "description": "SOC 值班台"}
          ]
        }
    """

    def __init__(self, principals: Optional[dict[str, ApiPrincipal]] = None):
        self._principals: dict[str, ApiPrincipal] = dict(principals or {})
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
        if not self._principals:
            self._principals = _load_principals_from_env()
        for key, p in self._principals.items():
            if p.rate_limit > 0:
                self._buckets[key] = TokenBucket(p.rate_limit, p.burst)

    @property
    def enabled(self) -> bool:
        return bool(self._principals)

    def authenticate(self, headers) -> Optional[ApiPrincipal]:
        """从请求头解析调用方；失败返回 None。

        用 ``hmac.compare_digest`` 做常数时间比较，避免用响应时间探测 key。
        """
        presented = _extract_key(headers)
        if not presented:
            return None
        for key, principal in self._principals.items():
            if hmac.compare_digest(key, presented):
                return principal
        return None

    def check_rate(self, principal: ApiPrincipal) -> tuple[bool, float]:
        """对该调用方做限流判定。"""
        with self._lock:
            bucket = None
            for key, p in self._principals.items():
                if p is principal:
                    bucket = self._buckets.get(key)
                    break
        if bucket is None:
            return True, 0.0
        return bucket.allow()


def _extract_key(headers) -> str:
    """从 Authorization: Bearer / X-API-Key 取出 key。"""
    auth = (headers.get("Authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (headers.get("X-API-Key") or "").strip()


def _principal_from_dict(d: dict) -> tuple[str, ApiPrincipal]:
    key = str(d.get("key") or "").strip()
    tenants = d.get("tenants")
    devices = d.get("devices")
    acl = None
    if tenants or devices:
        acl = ACLPolicy(
            tenants=set(tenants) if tenants else None,
            devices=set(devices) if devices else None,
            restrict_unlabeled=bool(d.get("restrict_unlabeled", False)),
        )
    principal = ApiPrincipal(
        key_id=str(d.get("id") or "anonymous"),
        acl=acl,
        rate_limit=float(d.get("rate_limit") or 0.0),
        burst=int(d.get("burst") or 0),
        description=str(d.get("description") or ""),
    )
    return key, principal


def _load_principals_from_env() -> dict[str, ApiPrincipal]:
    """从环境变量加载 key 表。"""
    out: dict[str, ApiPrincipal] = {}

    path = os.environ.get("EDR_RAG_API_KEYS_FILE", "").strip()
    if path:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            data = {}
        for entry in (data.get("keys") or []):
            if not isinstance(entry, dict):
                continue
            key, principal = _principal_from_dict(entry)
            if key:
                out[key] = principal

    single = os.environ.get("EDR_RAG_API_KEY", "").strip()
    if single:
        out.setdefault(
            single,
            ApiPrincipal(
                key_id="default",
                rate_limit=float(os.environ.get("EDR_RAG_RATE_LIMIT", "0") or 0),
                description="EDR_RAG_API_KEY",
            ),
        )
    return out
