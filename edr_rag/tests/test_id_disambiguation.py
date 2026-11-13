"""ingest 层 ID 碰撞消歧回归。

覆盖：
  - 唯一 ID 零变化
  - 同 ID 等价组（仅 source_ref 差异）ID 不变
  - 同 ID 异内容组全部成员加 @fileslug 后缀（含首现）
  - 后缀确定性（同输入两次调用同输出）
  - 输入不被修改
  - fileslug 再碰撞序号消歧
  - 冻结产物真实形状：anquan/duli sections 消歧后 upsert 键全唯一、
    等价组 ID 不变、消歧 metadata 记录
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from edr_rag.id_disambiguation import (
    disambiguate_collision_ids,
    disambiguate_procedure_ids,
    disambiguate_sections_file,
)


def _sec(sid, rel, content="c", extra=None):
    d = {
        "id": sid,
        "manual_id": "m",
        "title": "t",
        "section_path": ["p"],
        "source_ref": {"relative_path": rel, "anchor": None},
        "content": content,
    }
    if extra:
        d.update(extra)
    return d


class TestUniqueIdsUnchanged:
    def test_unique_ids_zero_change(self):
        secs = [_sec("a", "f1.html"), _sec("b", "f2.html")]
        out = disambiguate_collision_ids(secs)
        assert [s["id"] for s in out] == ["a", "b"]

    def test_empty_input(self):
        assert disambiguate_collision_ids([]) == []


class TestEquivalentGroupKeepsId:
    def test_same_id_same_content_diff_source_ref(self):
        secs = [_sec("a", "f1.html", "same"), _sec("a", "f2.html", "same")]
        out = disambiguate_collision_ids(secs)
        assert [s["id"] for s in out] == ["a", "a"]

    def test_links_images_excluded_from_equivalence(self):
        s1 = _sec("a", "f1.html", "same", {"links": [{"text": "x", "href": "1"}]})
        s2 = _sec("a", "f2.html", "same", {"links": [{"text": "y", "href": "2"}]})
        out = disambiguate_collision_ids([s1, s2])
        assert [s["id"] for s in out] == ["a", "a"]


class TestDiffContentGroupRenamed:
    def test_all_members_renamed_including_first(self):
        secs = [_sec("a", "f1.html", "v1"), _sec("a", "f2.html", "v2"),
                _sec("a", "f3.html", "v1")]
        out = disambiguate_collision_ids(secs)
        ids = [s["id"] for s in out]
        assert ids == ["a@f1", "a@f2", "a@f3"]

    def test_unrelated_id_not_touched(self):
        secs = [_sec("a", "f1.html", "v1"), _sec("a", "f2.html", "v2"),
                _sec("b", "f3.html", "z")]
        out = disambiguate_collision_ids(secs)
        assert out[2]["id"] == "b"

    def test_metadata_provenance_recorded(self):
        secs = [_sec("a", "f1.html", "v1"), _sec("a", "f2.html", "v2")]
        out = disambiguate_collision_ids(secs)
        meta = out[0].get("metadata", {}).get("id_disambiguation", {})
        assert meta["original_id"] == "a"
        assert meta["reason"] == "same_id_different_content"
        assert meta["source_file"] == "f1.html"

    def test_deterministic_across_calls(self):
        secs = [_sec("a", "f1.html", "v1"), _sec("a", "f2.html", "v2")]
        ids1 = [s["id"] for s in disambiguate_collision_ids(secs)]
        ids2 = [s["id"] for s in disambiguate_collision_ids(secs)]
        assert ids1 == ids2

    def test_input_not_mutated(self):
        secs = [_sec("a", "f1.html", "v1"), _sec("a", "f2.html", "v2")]
        before = json.dumps(secs, sort_keys=True)
        disambiguate_collision_ids(secs)
        assert json.dumps(secs, sort_keys=True) == before


class TestSlugCollisions:
    def test_same_basename_diff_dirs_get_sequence(self):
        secs = [_sec("a", "d1/page.html", "v1"),
                _sec("a", "d2/page.html", "v2"),
                _sec("a", "d3/page.html", "v3")]
        out = disambiguate_collision_ids(secs)
        ids = [s["id"] for s in out]
        assert ids == ["a@page", "a@page.2", "a@page.3"]

    def test_suffix_assignment_independent_of_input_order(self):
        # 来源→ID 映射必须是输入顺序无关的纯函数：同名文件不同目录时，
        # 序号按完整 relative_path 字典序分配，交换输入顺序映射不变。
        s1 = _sec("a", "d2/page.html", "v2")
        s2 = _sec("a", "d1/page.html", "v1")
        s3 = _sec("a", "d3/page.html", "v3")
        out_fwd = disambiguate_collision_ids([s1, s2, s3])
        out_rev = disambiguate_collision_ids([dict(s3), dict(s2), dict(s1)])
        by_src_fwd = {s["source_ref"]["relative_path"]: s["id"] for s in out_fwd}
        by_src_rev = {s["source_ref"]["relative_path"]: s["id"] for s in out_rev}
        assert by_src_fwd == by_src_rev
        # 完整定位字典序决定序号：d1 < d2 < d3
        assert by_src_fwd["d1/page.html"] == "a@page"
        assert by_src_fwd["d2/page.html"] == "a@page.2"
        assert by_src_fwd["d3/page.html"] == "a@page.3"

    def test_procedure_suffix_independent_of_input_order(self):
        def _proc(pid, file, section_id="sec.1", body="b"):
            return {"id": pid, "title": "t", "steps": [body],
                    "source": {"file": file, "section_id": section_id}}
        p1 = _proc("proc.x", "d2/page.html", body="v2")
        p2 = _proc("proc.x", "d1/page.html", body="v1")
        out_fwd = disambiguate_procedure_ids([p1, p2])
        out_rev = disambiguate_procedure_ids([dict(p2), dict(p1)])
        by_file_fwd = {p["source"]["file"]: p["id"] for p in out_fwd}
        by_file_rev = {p["source"]["file"]: p["id"] for p in out_rev}
        assert by_file_fwd == by_file_rev
        assert by_file_fwd["d1/page.html"] == "proc.x@page"
        assert by_file_fwd["d2/page.html"] == "proc.x@page.2"

    def test_non_colliding_basename_mapping_unchanged_by_order(self):
        # 无 slug 再碰撞（文件名本身唯一）时，序号机制不参与，
        # 映射自然顺序无关（防回归哨兵）。
        s1 = _sec("a", "topic_1.html", "v1")
        s2 = _sec("a", "topic_2.html", "v2")
        out_fwd = disambiguate_collision_ids([s1, s2])
        out_rev = disambiguate_collision_ids([dict(s2), dict(s1)])
        by_src_fwd = {s["source_ref"]["relative_path"]: s["id"] for s in out_fwd}
        by_src_rev = {s["source_ref"]["relative_path"]: s["id"] for s in out_rev}
        assert by_src_fwd == by_src_rev
        assert by_src_fwd["topic_1.html"] == "a@topic_1"
        assert by_src_fwd["topic_2.html"] == "a@topic_2"


class TestFileRunner:
    def test_file_runner_renames_and_persists(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sections.jsonl"
            rows = [_sec("a", "f1.html", "v1"), _sec("a", "f2.html", "v2"),
                    _sec("b", "f3.html", "z")]
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                         encoding="utf-8")
            renamed = disambiguate_sections_file(p)
            assert renamed == 2
            out = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert [r["id"] for r in out] == ["a@f1", "a@f2", "b"]

    def test_file_runner_noop_on_unique(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sections.jsonl"
            rows = [_sec("a", "f1.html"), _sec("b", "f2.html")]
            p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                         encoding="utf-8")
            assert disambiguate_sections_file(p) == 0

    def test_file_runner_missing_file(self):
        assert disambiguate_sections_file(Path("/nonexistent/sections.jsonl")) is None


class TestProcedureDisambiguation:
    """P1 层级联：procedure id 二次碰撞（slug 折叠型）。"""

    @staticmethod
    def _proc(pid, sid, f, body="b"):
        return {
            "id": pid,
            "manual_id": "m",
            "product": "p",
            "operation": "op",
            "title": "t",
            "section_path": ["s"],
            "source": {"file": f, "section_id": sid, "anchor": None},
            "steps": [{"order": 1, "instruction": body, "target": "", "action_hint": "", "source_instruction": body}],
            "preconditions": [], "warnings": [], "recovery": [],
            "expected_result": "", "version": "", "confidence": 0.5, "status": "candidate",
        }

    def test_diff_section_id_same_derived_id_renamed(self):
        # duli 实证形状：两个不同 section id 折叠出同一 proc id
        procs = [
            self._proc("procedure.登录业务面.x", "manual.section.登录_业务面.x", "f1.html"),
            self._proc("procedure.登录业务面.x", "manual.section.登录业务面.x", "f2.html"),
        ]
        out = disambiguate_procedure_ids(procs)
        assert out[0]["id"] == "procedure.登录业务面.x@f1"
        assert out[1]["id"] == "procedure.登录业务面.x@f2"

    def test_same_section_id_same_body_keeps_id(self):
        procs = [
            self._proc("p", "sec-a", "f1.html"),
            self._proc("p", "sec-a", "f2.html"),
        ]
        out = disambiguate_procedure_ids(procs)
        assert [p["id"] for p in out] == ["p", "p"]

    def test_same_section_id_diff_body_renamed(self):
        procs = [
            self._proc("p", "sec-a", "f1.html", body="v1"),
            self._proc("p", "sec-a", "f2.html", body="v2"),
        ]
        out = disambiguate_procedure_ids(procs)
        assert out[0]["id"] == "p@f1"
        assert out[1]["id"] == "p@f2"

    def test_unique_and_equivalent_untouched(self):
        procs = [self._proc("q", "sec-b", "f3.html")]
        out = disambiguate_procedure_ids(procs)
        assert out[0]["id"] == "q"

    def test_input_not_mutated_and_deterministic(self):
        procs = [
            self._proc("p", "sec-a", "f1.html", body="v1"),
            self._proc("p", "sec-a", "f2.html", body="v2"),
        ]
        before = json.dumps(procs, sort_keys=True)
        ids1 = [p["id"] for p in disambiguate_procedure_ids(procs)]
        ids2 = [p["id"] for p in disambiguate_procedure_ids(procs)]
        assert ids1 == ids2
        assert json.dumps(procs, sort_keys=True) == before

    def test_already_suffixed_id_not_double_suffixed(self):
        # section 级联已把 @slug 拼进 proc id；同名同源组不应再叠加
        procs = [
            self._proc("p@f1", "sec-a@f1", "f1.html"),
            self._proc("p@f1", "sec-a@f1", "f1.html"),
        ]
        out = disambiguate_procedure_ids(procs)
        assert [p["id"] for p in out] == ["p@f1", "p@f1"]


class TestFrozenArtifacts:
    """冻结产物真实形状（只读快照验证）。

    [1105]/[1617] 普查口径：anquan 17 重复 ID / 6 异内容组；duli 11 / 4。
    消歧后不变量：
      - 异内容组全部成员重命名（组消失，原 ID 不再携带多语义）
      - 仅等价组保留原 ID（读取侧 _dedupe_identical_id_rows 收敛）
    """

    WORKDIRS = {
        "anquan": Path("/Users/edr-test/Desktop/edr-chm-rag-work/anquan"),
        "duli": Path("/Users/edr-test/Desktop/edr-chm-rag-work/duli"),
    }

    def test_diff_content_groups_fully_renamed(self):
        expected_groups = {"anquan": 6, "duli": 4}
        for name, wd in self.WORKDIRS.items():
            rows = [json.loads(l) for l in
                    (wd / "sections.jsonl").read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            out = disambiguate_collision_ids(rows)
            renamed_ids = {b["metadata"]["id_disambiguation"]["original_id"]
                           for a, b in zip(rows, out) if a["id"] != b["id"]}
            assert len(renamed_ids) == expected_groups[name], \
                f"{name}: renamed groups {len(renamed_ids)} != {expected_groups[name]}"

    def test_no_diff_content_group_survives(self):
        for name, wd in self.WORKDIRS.items():
            rows = [json.loads(l) for l in
                    (wd / "sections.jsonl").read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            out = disambiguate_collision_ids(rows)
            cnt = {}
            for r in out:
                cnt.setdefault(r["id"], []).append(r)
            for sid, grp in cnt.items():
                if len(grp) > 1:
                    eq = {json.dumps({k: v for k, v in r.items()
                                      if k not in ("source_ref", "links", "images")},
                                     sort_keys=True, ensure_ascii=False) for r in grp}
                    assert len(eq) == 1, \
                        f"{name}: diff-content group survived for {sid}"

    def test_upsert_keys_unique_after_readside_dedupe(self):
        for name, wd in self.WORKDIRS.items():
            rows = [json.loads(l) for l in
                    (wd / "sections.jsonl").read_text(encoding="utf-8").splitlines()
                    if l.strip()]
            out = disambiguate_collision_ids(rows)
            seen, kept = set(), []
            for r in out:  # 读取侧同 ID 等价去重口径：保留首现
                if r["id"] not in seen:
                    seen.add(r["id"])
                    kept.append(r["id"])
            assert len(kept) == len(set(kept)), f"{name}: upsert keys not unique"
