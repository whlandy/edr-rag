"""edr-rag HTTP 服务层 (P4 Phase G)。

P0–P3 只有 stdio MCP 与 CLI，没有可被内网其它系统调用的接口，也没有鉴权、
限流、健康检查。本包补上这一层，纯 stdlib 实现（内网离线部署不引入
fastapi/uvicorn 依赖栈）。

::

    edr-rag-serve --workdir ~/edr-chm-rag-work --port 8080

鉴权见 ``auth`` 模块；未配置任何 API key 时服务**拒绝启动**。
"""

from __future__ import annotations

from .auth import ApiPrincipal, Authenticator, TokenBucket
from .server import EdrRagService, ServiceError, main, make_server, serve

__all__ = [
    "ApiPrincipal",
    "Authenticator",
    "TokenBucket",
    "EdrRagService",
    "ServiceError",
    "make_server",
    "serve",
    "main",
]
