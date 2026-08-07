"""P1-5 EDR 查询评测包。

提供评测集 (JSON/JSONL) + 评测 runner，用于量化不同检索策略
(keyword / bm25 / hybrid) 的质量差异，覆盖:

  - Recall@K      召回率
  - MRR           首个命中的倒数排名均值
  - nDCG@K        归一化折损累计增益
  - metadata filter accuracy 过滤器准确率

主要入口:
  - edr_rag.eval.dataset: 评测集 schema / 加载 / 校验
  - edr_rag.eval.metrics: 各指标计算
  - edr_rag.eval.runner:  评测 runner (对比多策略)
  - edr_rag.eval.cli:     CLI (edr-rag-eval)

评测集与 runner 均无需任何可选 `rag` 依赖；hybrid 在缺依赖时由
RetrievalService 自动降级为 RRF 融合，runner 直接复用，主链路始终可用。
"""

from .dataset import EvalSet, EvalQuery, load_eval_set  # noqa: F401
from .metrics import (  # noqa: F401
    reciprocal_rank,
    recall_at_k,
    ndcg_at_k,
    filter_accuracy,
    query_metrics,
    aggregate_metrics,
)
from .generation import (  # noqa: F401
    citation_accuracy,
    high_risk_identify_rate,
    high_risk_flag_hit,
    query_generation_metrics,
    refusal_hit,
    refusal_rate,
)
from .tune import TuningResult, tune_rrf_weights  # noqa: F401

__all__ = [
    "EvalSet",
    "EvalQuery",
    "load_eval_set",
    "reciprocal_rank",
    "recall_at_k",
    "ndcg_at_k",
    "filter_accuracy",
    "query_metrics",
    "aggregate_metrics",
    "citation_accuracy",
    "refusal_hit",
    "refusal_rate",
    "high_risk_flag_hit",
    "high_risk_identify_rate",
    "query_generation_metrics",
    "tune_rrf_weights",
    "TuningResult",
]
