"""P1-5 评测 runner 集成测试。

构造一个小型 workdir (actions/sections/procedures/chunks)，配一个评测集，
验证 runner:
  - 正确调用 RetrievalService 检索并计算各指标
  - 支持 keyword / bm25 / hybrid 多策略对比
  - metadata filter 用例的 filter_accuracy 计算
  - 缺依赖时 hybrid 优雅降级 (RRF 融合) 不报错
  - 同义改写放大评测规模
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.eval.dataset import EvalQuery, EvalSet, load_eval_set
from edr_rag.eval.runner import run_eval
from edr_rag.retrieval import RetrievalService


def _write_workdir(root: Path):
    actions = [
        {
            "action_id": "network.configure_proxy",
            "product": "EDR",
            "module": "network",
            "title": "配置代理服务器",
            "status": "approved",
            "confidence": 0.9,
            "risk": "medium",
            "requires_confirmation": True,
            "preconditions": ["已安装客户端"],
            "procedure": [{"order": 1, "action": "click", "target": "代理"}],
            "success_criteria": ["代理生效"],
            "recovery": [],
        },
        {
            "action_id": "system.reset_system",
            "product": "EDR",
            "module": "system",
            "title": "重置系统",
            "status": "needs_review",
            "confidence": 0.8,
            "risk": "high",
            "requires_confirmation": True,
            "preconditions": ["管理员权限"],
            "procedure": [{"order": 1, "action": "click", "target": "重置"}],
            "success_criteria": [],
            "recovery": ["恢复备份"],
        },
    ]
    (root / "actions.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in actions) + "\n",
        encoding="utf-8",
    )
    (root / "sections.jsonl").write_text(
        json.dumps({
            "id": "section.network.proxy",
            "title": "配置代理服务器",
            "content": "本节介绍如何配置代理，需要网络设置。",
            "section_path": ["网络", "代理"],
            "source": {"file": "network/proxy.html"},
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "procedures.jsonl").write_text(
        json.dumps({
            "id": "procedure.network.configure_proxy",
            "manual_id": "m1",
            "product": "EDR",
            "operation": "配置代理",
            "title": "配置代理服务器",
            "steps": [
                {"order": 1, "instruction": "打开网络设置"},
                {"order": 2, "instruction": "输入代理地址"},
            ],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    chunks = [
        {"id": "node.chunk.step.网络.代理.1", "title": "打开网络设置",
         "text": "打开网络设置选择代理"},
        {"id": "node.chunk.step.网络.代理.3", "title": "输入代理地址",
         "text": "输入代理地址并确认"},
    ]
    (root / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    _write_workdir(tmp_path)
    return tmp_path


def _default_eval_set():
    return EvalSet(
        name="demo",
        queries=[
            EvalQuery(
                qid="q1", corpus_type="actions", top_k=5,
                query="配置代理服务器", relevant=["network.configure_proxy"],
                paraphrases=["怎么设置代理", "代理怎么配"],
            ),
            EvalQuery(
                qid="q2", corpus_type="actions", top_k=5,
                query="重置系统", relevant=["system.reset_system"],
            ),
            EvalQuery(
                qid="q3", corpus_type="procedures", top_k=5,
                query="配置代理的步骤", relevant=["procedure.network.configure_proxy"],
            ),
            EvalQuery(
                qid="q4", corpus_type="sections", top_k=5,
                query="配置代理服务器", relevant=["section.network.proxy"],
            ),
            # metadata filter 用例：只保留 approved 动作，期待命中配置代理
            EvalQuery(
                qid="q5", corpus_type="actions", top_k=5,
                query="配置", relevant=["network.configure_proxy"],
                filter={"status": "approved"},
                filter_expected=["network.configure_proxy"],
            ),
        ],
    )


def test_run_eval_produces_rows_for_each_strategy(workdir):
    evalset = _default_eval_set()
    report = run_eval(workdir, evalset, strategies=("keyword", "bm25"))
    # 5 主查询 × 2 策略 = 10 行
    assert len(report.rows) == 10
    assert {r.strategy for r in report.rows} == {"keyword", "bm25"}
    summary = report.summary()
    assert set(summary.keys()) == {"keyword", "bm25"}
    assert "recall@k" in summary["keyword"]


def test_run_eval_keyword_recalls_proxy_action(workdir):
    evalset = _default_eval_set()
    report = run_eval(workdir, evalset, strategies=("keyword",))
    rows = {r.qid: r for r in report.rows}
    assert rows["q1"].ranked_ids[0] == "network.configure_proxy"
    assert rows["q1"].metrics["recall@k"] == 1.0
    assert rows["q1"].metrics["mrr"] == 1.0


def test_metadata_filter_accuracy_computed(workdir):
    evalset = _default_eval_set()
    report = run_eval(workdir, evalset, strategies=("keyword",))
    rows = {r.qid: r for r in report.rows}
    q5 = rows["q5"]
    assert q5.filter_returned == ["network.configure_proxy"]
    assert q5.metrics["filter_accuracy"] == 1.0


def test_hybrid_strategy_runs_without_optional_rags(workdir):
    # 缺 RAG 依赖时 hybrid 应降为 RRF 融合，不报错，仍有结果
    evalset = _default_eval_set()
    report = run_eval(workdir, evalset, strategies=("hybrid",))
    assert report.rows
    rows = {r.qid: r for r in report.rows}
    assert rows["q1"].ranked_ids[0] == "network.configure_proxy"


def test_use_paraphrases_expands_variants(workdir):
    evalset = _default_eval_set()
    base = run_eval(workdir, evalset, strategies=("keyword",), use_paraphrases=False)
    exp = run_eval(workdir, evalset, strategies=("keyword",), use_paraphrases=True)
    assert len(exp.rows) == 5 + 2  # q1 有 2 条改写
    assert len(exp.rows) > len(base.rows)
    qids = [r.qid for r in exp.rows]
    assert "q1#p1" in qids and "q1#p2" in qids


def test_run_eval_empty_strategies_returns_no_rows(workdir):
    evalset = _default_eval_set()
    report = run_eval(workdir, evalset, strategies=())
    assert report.rows == []


def test_reuse_provided_service(workdir):
    svc = RetrievalService(workdir)
    evalset = _default_eval_set()
    report = run_eval(workdir, evalset, strategies=("keyword",), service=svc)
    assert report.rows


def test_load_eval_set_jsonl(workdir):
    data = workdir / "mini.jsonl"
    data.write_text(
        json.dumps({
            "qid": "m1", "corpus_type": "actions",
            "query": "配置代理服务器", "relevant": ["network.configure_proxy"],
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    evalset = load_eval_set(data)
    assert len(evalset.queries) == 1
    assert evalset.queries[0].qid == "m1"


def test_default_dataset_ships_and_validates():
    here = Path(__file__).parent
    data_path = here / "data" / "eval_set_default.jsonl"
    assert data_path.exists()
    evalset = load_eval_set(data_path)
    # 达到 100+ 条变体 (desc 要求同义改写可凑规模)
    assert evalset.variant_total >= 100
    # 覆盖多个 corpus
    corpora = {q.corpus_type for q in evalset.queries}
    assert {"actions", "sections", "procedures", "chunks"} <= corpora


# ---------- P2-6: generation 维度 ----------


def _gen_eval_set():
    """带生成维度标注的评测集。

    - qr1: 应拒答（应作答的隐私查询，无足够证据）
    - qr2: 普通可作答，但需识别高风险
    - qr3: 普通可作答，无高风险
    """
    return EvalSet(
        name="gen-demo",
        queries=[
            EvalQuery(
                qid="qr1", corpus_type="actions", top_k=5,
                query="怎么查看同事的私有文件", relevant=[],
                expected_refusal=True,
            ),
            EvalQuery(
                qid="qr2", corpus_type="actions", top_k=5,
                query="重置系统", relevant=["system.reset_system"],
                high_risk=True,
            ),
            EvalQuery(
                qid="qr3", corpus_type="actions", top_k=5,
                query="配置代理服务器", relevant=["network.configure_proxy"],
                expected_refusal=False,
                high_risk=False,
            ),
        ],
    )


def test_run_eval_populates_generation_metrics(workdir):
    evalset = _gen_eval_set()
    report = run_eval(workdir, evalset, strategies=("keyword",))
    rows = {r.qid: r for r in report.rows}
    for r in report.rows:
        # 有检索命中即有答案；至少引用正确率存在。
        assert "citation_accuracy" in r.generation_metrics
    # qr2 命中重置系统(high risk, requires_confirmation=True) -> 高风识别应准确
    assert rows["qr2"].answer is not None
    assert rows["qr2"].answer.requires_confirmation is True
    assert rows["qr2"].generation_metrics.get("high_risk_correct") == 1.0


def test_generation_summary_shape(workdir):
    evalset = _gen_eval_set()
    report = run_eval(workdir, evalset, strategies=("keyword", "bm25"))
    gs = report.generation_summary()
    assert set(gs.keys()) == {"keyword", "bm25"}
    for metrics in gs.values():
        assert {"citation_accuracy", "refusal_recall", "false_refusal", "high_risk"} <= set(metrics)


def test_generation_summary_refusal_metrics(workdir):
    evalset = _gen_eval_set()
    report = run_eval(workdir, evalset, strategies=("keyword",))
    gs = report.generation_summary()
    kw = gs["keyword"]
    # qr1 应拒答且无相关证据 -> 应被拒答；qr3 应作答且命中 -> 不应被拒答。
    assert kw["refusal_recall"] == 1.0
    assert kw["false_refusal"] == 0.0
    # 两个应作答样本 qr2/qr3 命中真实动作 -> 引用正确率高。
    assert kw["citation_accuracy"] >= 0.5
    assert kw["high_risk"] == 1.0  # 唯一应识别高风险的 qr2 被正确识别


def test_run_eval_llm_insufficient_evidence_marks_refusal(workdir):
    """llm 注入（缺口2）+ LLM 判证据不足拒答（缺口1）端到端：

    组装模式拒答失效（[20261111-2057] 实证 30/30 answered）；注入会抛
    LLMInsufficientEvidenceError 的 llm 后，expected_refusal 的查询应记
    refused 且 refusal_correct 生效。
    """
    from edr_rag.llm.base import LLMInsufficientEvidenceError

    def refusing_llm(q, c):
        # 对应拒答标注的查询判证据不足；其余照常返回生成文本
        if "同事" in q or "私有文件" in q:
            raise LLMInsufficientEvidenceError("insufficient")
        return f"LLM回答：{q}"

    evalset = _gen_eval_set()
    report = run_eval(workdir, evalset, strategies=("keyword",), llm=refusing_llm)
    rows = {r.qid: r for r in report.rows}
    # qr1 应拒答 -> LLM 拒答生效
    assert rows["qr1"].answer.answer_status == "refused"
    assert rows["qr1"].generation_metrics.get("refusal_correct") == 1.0
    # qr2/qr3 应作答 -> LLM 生成文本被采用（qr2 未标 expected_refusal，
    # 不产生 refusal_correct 指标——指标只对有标注的查询计算）
    assert rows["qr2"].answer.answer_status == "answered"
    assert rows["qr2"].answer.generated_by == "llm"
    assert "refusal_correct" not in rows["qr2"].generation_metrics


def test_run_eval_default_no_llm_unchanged(workdir):
    """缺省 llm=None 时行为与既有组装路径完全一致（回归保护）。"""
    evalset = _gen_eval_set()
    report = run_eval(workdir, evalset, strategies=("keyword",))
    rows = {r.qid: r for r in report.rows}
    assert rows["qr2"].answer.generated_by == "assemble"


def test_load_eval_set_parses_annotation_bools(workdir):
    data = workdir / "gen.jsonl"
    data.write_text(
        json.dumps({
            "qid": "g1", "corpus_type": "actions",
            "query": "重置系统", "relevant": ["system.reset_system"],
            "expected_refusal": False, "high_risk": True,
        }, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    evalset = load_eval_set(data)
    q = evalset.queries[0]
    assert q.expected_refusal is False
    assert q.high_risk is True
