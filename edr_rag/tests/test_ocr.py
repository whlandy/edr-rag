"""Tests for edr_rag.ocr (P3-4 图片 OCR/多模态截图检索).

Coverage:
  - ocr_available() reflects real backend (pytesseract+tesseract present here)
  - ocr_image() returns text for a real synthetic image with words
  - ocr_image() on missing/invalid input returns "" (never raises)
  - run_ocr(): 写入 assets.jsonl 的 ocr_text + 产出 ocr.jsonl 可检索产物
  - run_ocr(): 幂等 —— 已有 ocr_text 不重复识别（除非 force）
  - run_ocr(): 无资产/非图片资产 → 干净返回
  - retrieval: corpus_type="images" 能命中 ocr.jsonl 里的截图文字
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest  # noqa: E402

from edr_rag import ocr as ocr_mod  # noqa: E402


# ---------------------------------------------------------------------------
# P4: 依赖真实 OCR 引擎的用例改为 skipif。
#
# 旧实现在缺 tesseract / rapidocr 时直接 assert 失败——OCR 引擎是 pyproject
# 里的**可选** extras，缺它不是缺陷。CI 跑在干净环境里会永远红 6 个，
# 于是「测试是绿的」这条信号就废了。
#
# 纯逻辑用例（缺失文件、非图片、幂等语义、产物结构）不依赖引擎，照常运行。
# ---------------------------------------------------------------------------

requires_ocr = pytest.mark.skipif(
    not ocr_mod.ocr_available(),
    reason="OCR engine (tesseract / rapidocr) not installed; optional `ocr` extras",
)

@contextmanager
def _tmp():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def _make_text_png(path: Path, text: str) -> Path:
    """用 Pillow 生成一张含文字的 PNG（OCR 能识别出文字）。"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (360, 120), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except Exception:
        font = ImageFont.load_default()
    draw.text((20, 40), text, fill="black", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


def _write_assets(workdir: Path, records: list[dict]) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    p = workdir / "assets.jsonl"
    p.write_text(lines, encoding="utf-8")
    return p


@requires_ocr
def test_ocr_available_reflects_backend():
    # 走到这里说明 skipif 判定引擎可用，探测函数必须自洽。
    assert ocr_mod.ocr_available() is True
    assert ocr_mod.get_ocr_engine() is not None
    print(f"[PASS] ocr_available=True, engine={ocr_mod.get_ocr_engine() and 'detected'}")


@requires_ocr
def test_ocr_image_real_text():
    with _tmp() as tmp:
        png = _make_text_png(tmp / "shot.png", "Proxy Configuration")
        text = ocr_mod.ocr_image(png, lang="eng")
        assert "Proxy" in text, f"OCR did not extract 'Proxy' from image, got: {text!r}"
    print(f"[PASS] ocr_image extracted text: {text!r}")


def test_ocr_image_graceful_on_missing():
    # 不存在的文件 / 非图片 → 空串，不抛异常
    assert ocr_mod.ocr_image(Path("/nonexistent/nowhere.png")) == ""
    print("[PASS] ocr_image on missing file returns ''")


@requires_ocr
def test_run_ocr_writes_artifacts():
    with _tmp() as tmp:
        png = _make_text_png(tmp / "png" / "proxy.png", "Enable SSL Proxy")
        _write_assets(tmp, [
            {
                "id": "asset.images.proxy.png",
                "manual_id": "m",
                "type": "image",
                "path": "png/proxy.png",
                "absolute_path": str(png),
                "referenced_by": ["network/proxy.html"],
            },
            {
                "id": "asset.nonimage.css",
                "manual_id": "m",
                "type": "css",
                "path": "site/s.css",
                "absolute_path": str(tmp / "site" / "s.css"),
                "referenced_by": [],
            },
        ])
        summary = ocr_mod.run_ocr(tmp)
        assert summary.total_images == 1
        assert summary.ocr_succeeded == 1, summary
        # assets.jsonl 写回 ocr_text
        recs = [
            json.loads(l)
            for l in (tmp / "assets.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        image = next(r for r in recs if r["type"] == "image")
        assert image["ocr_text"] and "SSL" in image["ocr_text"], image
        # ocr.jsonl 产出
        assert (tmp / "ocr.jsonl").exists()
        ocr_recs = [
            json.loads(l)
            for l in (tmp / "ocr.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        assert len(ocr_recs) == 1
        assert "SSL" in ocr_recs[0]["text"]
        assert ocr_recs[0]["source"]["file"] == "png/proxy.png"
    print("[PASS] run_ocr writes ocr_text into assets.jsonl + ocr.jsonl")


@requires_ocr
def test_run_ocr_idempotent():
    with _tmp() as tmp:
        png = _make_text_png(tmp / "png" / "a.png", "Delta Sync")
        _write_assets(tmp, [
            {
                "id": "asset.images.a.png",
                "manual_id": "m",
                "type": "image",
                "path": "png/a.png",
                "absolute_path": str(png),
                "referenced_by": [],
            },
        ])
        s1 = ocr_mod.run_ocr(tmp)
        assert s1.ocr_succeeded == 1
        # 第二次跑：已有 ocr_text → 复用，不再新增 succeeded（幂等）
        s2 = ocr_mod.run_ocr(tmp)
        assert s2.ocr_succeeded == 1, s2
        assert s2.empty_or_skipped == 0, s2
        # force=True 会强制重识别（仍有引擎时 succeeded 仍为 1）
        s3 = ocr_mod.run_ocr(tmp, force=True)
        assert s3.ocr_succeeded == 1, s3
    print("[PASS] run_ocr idempotent rerun")


def test_run_ocr_no_assets():
    with _tmp() as tmp:
        tmp.mkdir(parents=True, exist_ok=True)
        summary = ocr_mod.run_ocr(tmp)
        assert summary.total_images == 0
        assert summary.ocr_jsonl == ""
        # 不应生成 ocr.jsonl / 不应崩溃
        assert not (tmp / "ocr.jsonl").exists()
    print("[PASS] run_ocr with no assets.jsonl returns clean empty summary")


def test_run_ocr_missing_image_file_empty_text():
    """引用了一个磁盘上不存在的图片：OCR 得空 text，但不崩溃，entry 仍写入 ocr.jsonl。"""
    with _tmp() as tmp:
        _write_assets(tmp, [
            {
                "id": "asset.images.gone.png",
                "manual_id": "m",
                "type": "image",
                "path": "gone.png",
                "absolute_path": str(tmp / "gone.png"),  # 不存在
                "referenced_by": [],
            },
        ])
        summary = ocr_mod.run_ocr(tmp)
        assert summary.total_images == 1
        assert summary.empty_or_skipped == 1, summary
        assert (tmp / "ocr.jsonl").exists()
    print("[PASS] run_ocr tolerates missing image file")


@requires_ocr
def test_retrieval_images_corpus():
    """corpus_type='images' 能命中 ocr.jsonl 里的截图文字。"""
    from edr_rag.retrieval import RetrievalService

    with _tmp() as tmp:
        png = _make_text_png(tmp / "png" / "diagram.png", "LED Indicator Blinks Three Times")
        _write_assets(tmp, [
            {
                "id": "asset.images.diagram.png",
                "manual_id": "m",
                "type": "image",
                "path": "png/diagram.png",
                "absolute_path": str(png),
                "referenced_by": ["led/status.html"],
            },
        ])
        ocr_mod.run_ocr(tmp)
        svc = RetrievalService(workdir=str(tmp))
        hits = svc.search("LED Indicator", corpus_type="images", top_k=5)
        assert hits, "expected images corpus to hit ocr.jsonl text"
        assert hits[0]["type"] == "image"
        assert "diagram.png" in hits[0].get("id", "")
    print("[PASS] retrieval corpus_type='images' hits OCR text")


@requires_ocr
def test_search_images_mcp_backend():
    """MCP search_images 走 corpus_type='images'，可检索截图文字。"""
    from edr_rag.p3.server import EdrRagAgent

    with _tmp() as tmp:
        png = _make_text_png(tmp / "png" / "c.png", "Reset Button Labeled RST")
        _write_assets(tmp, [
            {
                "id": "asset.images.c.png",
                "manual_id": "m",
                "type": "image",
                "path": "png/c.png",
                "absolute_path": str(png),
                "referenced_by": ["hw/reset.html"],
            },
        ])
        ocr_mod.run_ocr(tmp)
        agent = EdrRagAgent(workdir=str(tmp))
        results = agent.search_images("RST", limit=5)
        assert results, "search_images should hit OCR text 'RST'"
    print("[PASS] MCP search_images returns OCR hits")

