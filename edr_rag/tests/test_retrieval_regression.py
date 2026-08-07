"""P0-7 核心检索回归测试套件。

为 edr-rag 的核心检索入口建立完整、可回归的测试覆盖，防止后续改动
破坏既有检索行为。覆盖对象:

  - EdrRagAPI.action_search / get_action   (P2 动作目录，最高优先级)
  - EdrRagAPI.procedure_search             (P1 流程，次优先级)
  - EdrRagAPI.manual_search                (P0 章节，兜底)
  - RetrievalService.planner_retrieve      (action -> procedure -> manual 编排)
  - JsonlCorpus.keyword_search / bm25_search  (底层 corpus 检索)

覆盖的检索行为 (对应 P0-7 desc):
  1. CJK 检索        —— 中文查询能召回中文内容，且命中文案在 title/snippet 命中。
  2. 风险/状态过滤    —— risk / status 单过滤与组合过滤都正确。
  3. 结果去重        —— 同一查询返回的结果中不存在重复 id。
  4. 空查询          —— 空串/纯空白查询返回空列表，不抛异常。
  5. 无结果          —— 明确不存在的查询返回空列表。
  6. limit 边界      —— 0 / 负值 / 超过语料总量 / 精确值 都符合预期。

测试数据统一由 _write_corpora 生成：6 个动作(不同 risk/status)、
1 个 procedure、1 个 section，保证过滤器与去重断言有足够的区分度。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from edr_rag.p1.search import JsonlCorpus
from edr_rag.p2.api import EdrRagAPI
from edr_rag.retrieval import RetrievalService

# ---------- 测试语料 ----------

# 6 个动作：risk 在 low/high 之间交替，status 在 approved/needs_review 交替，
# 提供足够的过滤区分度。title 含中文「配置代理」，可验证 CJK 检索。
_ACTIONS = [
    {
        "action_id": f"net.act_{i}",
        "product": "EDR",
        "module": "net",
        "title": f"配置代理服务器 {i}",
        "status": "approved" if i % 2 == 0 else "needs_review",
        "confidence": 0.9,
        "risk": "low" if i % 3 == 0 else "high",
        "requires_confirmation": True,
        "preconditions": [],
        "procedure": [{"order": 1, "action": "click", "target": "配置"}],
        "success_criteria": [],
        "recovery": [],
    }
    for i in range(6)
]

# 唯一一个与动作标题区分的问题：以步骤内容 (非标题) 区分，验证 procedure 检索。
_PROCEDURE = {
    "id": "procedure.net.act_0",
    "manual_id": "m1",
    "product": "EDR",
    "operation": "配置代理",
    "title": "配置代理服务器步骤说明",
    "text": "打开网络设置，输入代理服务器地址，点击保存。",
    "section_path": ["网络", "代理"],
    "source": {"file": "net/proxy.html"},
}

# 与 actions/procedure 都不同词的手册章节，用于 manual 兜底检索。
_SECTION = {
    "id": "s1",
    "title": "联通性故障排查",
    "content": "当设备无法上网时，请检查物理链路、IP 地址与 DNS 配置。",
    "section_path": ["故障排查", "网络"],
    "source": {"file": "troubleshoot/net.html"},
}


def _write_corpora(root: Path) -> Path:
    """把固定测试语料写入 root 下的 actions/procedures/sections.jsonl。"""
    (root / "actions.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in _ACTIONS) + "\n",
        encoding="utf-8",
    )
    (root / "procedures.jsonl").write_text(
        json.dumps(_PROCEDURE, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    (root / "sections.jsonl").write_text(
        json.dumps(_SECTION, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    return root


@pytest.fixture
def api(tmp_path: Path) -> EdrRagAPI:
    """带标准语料的 API 实例 (懒加载)。"""
    _write_corpora(tmp_path)
    return EdrRagAPI(tmp_path)


@pytest.fixture
def service(tmp_path: Path) -> RetrievalService:
    """带标准语料的 RetrievalService 实例。"""
    _write_corpora(tmp_path)
    return RetrievalService(tmp_path)


# ---------- 1. CJK 检索 ----------

def test_action_search_cjk_recalls_chinese(api):
    res = api.action_search("配置代理")
    assert res, "中文查询应能召回中文动作"
    assert any("配置代理" in r["title"] for r in res)


def test_action_search_cjk_by_partial_token(api):
    # 中文按 unigram 分词，"配置" 应能命中所有 6 个动作 (limit 需足够大)。
    res = api.action_search("配置", limit=100)
    assert len(res) == len(_ACTIONS)


def test_procedure_search_cjk_by_step_content(api):
    # 以"步骤内容"(非标题关键词)查 procedure —— 验证 procedure 检索按全文召回。
    res = api.procedure_search("代理服务器地址")
    assert res
    assert res[0]["id"] == "procedure.net.act_0"


def test_manual_search_cjk_recalls_section(api):
    # "无法上网" 仅出现在 section 正文中 —— 验证 manual 检索按正文召回。
    res = api.manual_search("无法上网")
    assert res
    assert res[0]["id"] == "s1"


def test_planner_retrieve_cjk_action_first(service):
    res = service.planner_retrieve("配置")
    assert res["actions"], "动作命中时应走第一优先级"
    assert res["procedures"] == [] and res["sections"] == []


def test_planner_retrieve_falls_back_to_section(service):
    # "故障排查" 的四个 unigram (故/障/排/查) 只存在于 section，
    # actions/procedures 均不含 -> 逐级降级到 manual 层。
    res = service.planner_retrieve("故障排查")
    assert not res["actions"]
    assert not res["procedures"]
    assert res["sections"]


# ---------- 2. 风险 / 状态过滤 ----------

def test_action_search_risk_filter(api):
    res = api.action_search("配置", risk_filter="low")
    assert res
    assert all(r["risk"] == "low" for r in res)
    # low 个数据实为 i%3==0 -> act_0, act_3 —— 恰好 2 个。
    assert len(res) == 2


def test_action_search_status_filter(api):
    res = api.action_search("配置", status_filter="approved")
    assert res
    assert all(r["status"] == "approved" for r in res)
    # approved 为偶数 i -> act_0..5 共 3 个。
    assert len(res) == 3


def test_action_search_combined_risk_status_filter(api):
    # risk=low 且 status=approved 的唯一交叠：act_0 (i%3==0 且 i%2==0)。
    res = api.action_search("配置", risk_filter="low", status_filter="approved")
    assert [r["id"] for r in res] == ["net.act_0"]


def test_action_search_filter_all_match_when_query_is_noop(api):
    # 空查询时即使带过滤也返回空 —— 过滤不会凭空造出结果。
    assert api.action_search("", risk_filter="low") == []
    assert api.action_search("", status_filter="approved") == []


def test_retrieval_service_filter_dict(service):
    # service 层 filters 字典与 API 层参数等价。
    res = service.search("配置", corpus_type="actions", filters={"risk": "high"})
    assert res and all(r["risk"] == "high" for r in res)


# ---------- 3. 结果去重 ----------

def _unique_ids(results) -> list[str]:
    return [r["id"] for r in results]


@pytest.mark.parametrize("method,query", [
    ("action_search", "配置"),
    ("procedure_search", "代理"),
    ("manual_search", "排查"),
])
def test_search_results_no_duplicate_ids(api, method, query):
    # 即便 limit 远超语料总量，返回结果 id 也不允许重复。
    res = getattr(api, method)(query, limit=100)
    ids = _unique_ids(res)
    assert len(ids) == len(set(ids)), f"{method} 返回了重复 id: {ids}"


def test_planner_retrieve_tiers_no_duplicate_within_tier(service):
    res = service.planner_retrieve("配置代理", limit=100)
    for tier in ("actions", "procedures", "sections"):
        ids = _unique_ids(res[tier])
        assert len(ids) == len(set(ids)), f"tier {tier} 出现重复 id"


def test_jsonl_corpus_keyword_no_duplicate():
    corpus = JsonlCorpus([
        {"id": "a", "title": "配置代理", "content": "配置代理服务器", "source": {"file": "a"}},
        {"id": "b", "title": "重置系统", "content": "重置系统", "source": {"file": "b"}},
    ])
    for strat_call in (corpus.keyword_search, corpus.bm25_search):
        res = strat_call("配置", limit=10)
        ids = [r["id"] for r in res]
        assert len(ids) == len(set(ids))


# ---------- 4. 空查询 ----------

@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_action_search_empty_query_returns_empty(api, query):
    assert api.action_search(query) == []


@pytest.mark.parametrize("fn", [
    lambda a: a.procedure_search(""),
    lambda a: a.manual_search(""),
    lambda a: a.action_search("   "),
])
def test_api_empty_query_never_raises(api, fn):
    # 空查询必须返回空列表，绝不抛异常。
    assert fn(api) == []


def test_planner_empty_query(service):
    res = service.planner_retrieve("")
    assert res["actions"] == []
    assert res["procedures"] == [] and res["sections"] == []


@pytest.mark.parametrize("method", [JsonlCorpus.keyword_search, JsonlCorpus.bm25_search])
def test_jsonl_corpus_empty_query(method):
    corpus = JsonlCorpus([{"id": "a", "title": "x", "content": "y"}])
    assert method(corpus, "") == []
    assert method(corpus, "   ") == []


# ---------- 5. 无结果 ----------
# 注意：中文按 unigram 分词后，任意中文单字都可能误命中语料中的常见字。
# 因此"无结果"用例统一使用语料中绝不出现的小写拉丁词 qxyz（tokenize 会
# 将其转为小写 token；语料文本为 raw 存储，不含该词），保证可靠地返回空。

def test_action_search_no_result(api):
    assert api.action_search("qxyz") == []


def test_procedure_search_no_result(api):
    assert api.procedure_search("qxyz") == []


def test_manual_search_no_result(api):
    assert api.manual_search("qxyz") == []


def test_planner_retrieve_no_result(service):
    res = service.planner_retrieve("qxyz")
    assert res["actions"] == []
    assert res["procedures"] == []
    assert res["sections"] == []


def test_jsonl_corpus_no_result():
    corpus = JsonlCorpus([{"id": "a", "title": "配置", "content": "代理"}])
    assert corpus.keyword_search("qxyz") == []
    assert corpus.bm25_search("qxyz") == []


# ---------- 6. limit 边界 ----------

def test_limit_zero_returns_empty(api):
    assert api.action_search("配置", limit=0) == []
    assert api.procedure_search("代理", limit=0) == []
    assert api.manual_search("排查", limit=0) == []


def test_limit_negative_clamped_to_empty(api):
    # 负 limit 不抛异常；服务层 clamp 到 0 => 空。
    assert api.action_search("配置", limit=-1) == []
    assert api.procedure_search("代理", limit=-1) == []
    assert api.manual_search("排查", limit=-1) == []


def test_limit_exact_and_above_corpus(api):
    # 精确值：恰好取到要求条数（数据不足时取全部）。
    assert len(api.action_search("配置", limit=2)) == 2
    # 超量：返回全部可用结果，不重复、不报错。
    res = api.action_search("配置", limit=100)
    assert len(res) == len(_ACTIONS)
    assert len(_unique_ids(res)) == len(_ACTIONS)


def test_limit_boundary_jsonl_corpus():
    corpus = JsonlCorpus([
        {"id": "a", "title": "配置", "content": "代理"},
        {"id": "b", "title": "配置", "content": "代理"},
        {"id": "c", "title": "配置", "content": "代理"},
    ])
    assert len(corpus.keyword_search("配置", limit=2)) == 2
    assert len(corpus.bm25_search("配置", limit=2)) == 2
    assert len(corpus.keyword_search("配置", limit=0)) == 0
    assert len(corpus.keyword_search("配置", limit=100)) == 3


def test_planner_retrieve_limit_boundary(service):
    # limit 透传到每层；动作全命中时应取 actions 层，不触发 fallback。
    res = service.planner_retrieve("配置弟弟", limit=2)
    assert len(res["actions"]) <= 2
    res = service.planner_retrieve("配置", limit=100)
    assert len(res["actions"]) == len(_ACTIONS)


# ---------- 7. 精确获取回归 ----------

def test_get_action_by_id_and_prefixed_id(api):
    full = api.get_action("net.act_0")
    assert full is not None and full["action_id"] == "net.act_0"
    # 兼容 'action.' / 'procedure.' 前缀。
    assert api.get_action("action.net.act_0") == full
    assert api.get_action("procedure.net.act_0") == full


def test_get_action_missing_returns_none(api):
    assert api.get_action("no.such.action") is None
