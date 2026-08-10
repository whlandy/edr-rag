"""P4 —— OCR 产物逻辑的**引擎无关**测试。

背景：``test_ocr.py`` 里依赖真实 tesseract / rapidocr 的用例现在是 skipif
(OCR 引擎是可选 extras，缺它不是缺陷)。但如果只有 skip，这条路径在干净的
CI 环境里就完全没有覆盖 —— 产物结构、幂等语义、降级行为都无人把守。

这里用一个假引擎（返回确定性文本）覆盖 ``run_ocr`` 的全部逻辑，不需要装
任何 OCR 依赖。真实引擎的识别质量由 test_ocr.py 在装了 extras 的环境里验证，
两者互补。
"""

from __future__ import annotations

import json

import pytest

from edr_rag import ocr as ocr_mod


@pytest.fixture(autouse=True)
def _reset_engine_cache():
    ocr_mod._ENGINE_CACHE = None
    yield
    ocr_mod._ENGINE_CACHE = None


@pytest.fixture
def fake_engine(monkeypatch):
    """装一个确定性假 OCR 引擎：识别结果 = 文件名的大写形式。"""
    calls: list[str] = []

    def _engine(path, lang):
        calls.append(str(path))
        return f"TEXT-{path.stem.upper()}"

    monkeypatch.setattr(ocr_mod, "get_ocr_engine", lambda: _engine)
    return calls


def _write_assets(workdir, records):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "assets.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def _image_asset(workdir, name="shot1"):
    """构造一条与 P0 产物同形状的图片资产记录 (type/path/absolute_path)。"""
    img = workdir / "png" / f"{name}.png"
    img.parent.mkdir(parents=True, exist_ok=True)
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)  # 内容不重要，引擎是假的
    return {
        "id": f"asset.images.{name}.png",
        "manual_id": "m",
        "type": "image",
        "path": f"png/{name}.png",
        "absolute_path": str(img),
        "referenced_by": ["network/proxy.html"],
    }


# ---------- 降级行为 ----------


def test_no_assets_file_returns_empty(tmp_path):
    summary = ocr_mod.run_ocr(tmp_path)
    assert summary.total_images == 0
    assert summary.ocr_succeeded == 0


def test_no_engine_skips_cleanly(tmp_path, monkeypatch):
    """无 OCR 引擎时干净跳过，不抛异常、不写半截产物。"""
    monkeypatch.setattr(ocr_mod, "get_ocr_engine", lambda: None)
    _write_assets(tmp_path, [_image_asset(tmp_path)])
    summary = ocr_mod.run_ocr(tmp_path)
    assert summary.ocr_succeeded == 0


def test_non_image_assets_ignored(tmp_path, fake_engine):
    _write_assets(tmp_path, [
        {"id": "asset.css", "type": "css", "path": "css/style.css"},
        {"id": "asset.js", "type": "js", "path": "js/app.js"},
    ])
    summary = ocr_mod.run_ocr(tmp_path)
    assert summary.total_images == 0
    assert fake_engine == [], "非图片资产不该送进 OCR"


# ---------- 产物结构 ----------


def test_writes_ocr_jsonl_with_searchable_shape(tmp_path, fake_engine):
    _write_assets(tmp_path, [_image_asset(tmp_path)])
    summary = ocr_mod.run_ocr(tmp_path)
    assert summary.ocr_succeeded == 1

    ocr_path = tmp_path / "ocr.jsonl"
    assert ocr_path.is_file()
    rows = [json.loads(ln) for ln in ocr_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(rows) == 1
    row = rows[0]
    # 供 RetrievalService corpus_type="images" 检索的最小契约
    for key in ("id", "text"):
        assert key in row, f"ocr.jsonl 缺少字段 {key}"
    assert "TEXT-SHOT1" in row["text"]


def test_ocr_text_written_back_to_assets(tmp_path, fake_engine):
    _write_assets(tmp_path, [_image_asset(tmp_path)])
    ocr_mod.run_ocr(tmp_path)
    rows = [
        json.loads(ln)
        for ln in (tmp_path / "assets.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert rows[0].get("ocr_text"), "assets.jsonl 没有回写 ocr_text"


# ---------- 幂等 ----------


def test_second_run_does_not_reocr(tmp_path, fake_engine):
    _write_assets(tmp_path, [_image_asset(tmp_path)])
    ocr_mod.run_ocr(tmp_path)
    first_calls = len(fake_engine)
    assert first_calls == 1

    ocr_mod.run_ocr(tmp_path)
    assert len(fake_engine) == first_calls, "已有 ocr_text 的资产被重复识别"


def test_force_reocrs(tmp_path, fake_engine):
    _write_assets(tmp_path, [_image_asset(tmp_path)])
    ocr_mod.run_ocr(tmp_path)
    ocr_mod.run_ocr(tmp_path, force=True)
    assert len(fake_engine) == 2, "force=True 应强制重新识别"


# ---------- 检索可用性 ----------


def test_images_corpus_is_searchable(tmp_path, fake_engine):
    """OCR 产物必须真的能被 corpus_type='images' 检索到。"""
    from edr_rag.retrieval import RetrievalService

    _write_assets(tmp_path, [_image_asset(tmp_path, "proxydialog")])
    ocr_mod.run_ocr(tmp_path)
    results = RetrievalService(tmp_path).search(
        "PROXYDIALOG", corpus_type="images", top_k=5
    )
    assert results, "images 语料没能命中 OCR 文本"


def test_ocr_disabled_by_env(tmp_path, monkeypatch):
    """EDR_RAG_OCR=0 时整条 OCR 链路关闭。"""
    monkeypatch.setenv("EDR_RAG_OCR", "0")
    ocr_mod._ENGINE_CACHE = None
    assert ocr_mod.get_ocr_engine() is None
    assert ocr_mod.ocr_available() is False
