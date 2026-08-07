"""P3-4 图片 OCR —— 把 CHM 内图片资源（截图）文本化，纳入检索。

设计目标（对应 P3-4 desc）:
  手册里大量操作说明以截图（PNG/JPG/GIF 等）形式呈现，这些信息在 P0/P1
  只被记录成 asset（type=image），正文不包含截图里的文字。本模块提供一条
  「可选依赖」的 OCR 通道:

  1. get_ocr_engine() / ocr_available()
     探测可用 OCR 后端。优先级:
       a. pytesseract + Pillow + 系统 tesseract 二进制（能输出中英文，
          需要 tessdata 含 chi_sim / eng）;
       b. rapidocr_onnxruntime（纯 Python，无系统依赖，自动下载 onnx 模型，
          内置中英文识别）。
     两个都缺时返回 None —— 主链路不报错，调用方（run_ocr）直接标记跳过。

  2. ocr_image(image_path, lang=...) -> str
     对单张图片做 OCR，返回识别文本（失败返回空字符串，绝不上抛）。

  3. run_ocr(workdir, ...) -> OcrSummary
     读取 <workdir>/assets.jsonl，对 type=="image" 的资产逐个 OCR，
     把结果:
       - 写回 assets.jsonl 每条记录的 ocr_text 字段;
       - 生成新的 <workdir>/ocr.jsonl 可检索产物（供 RetrievalService 的
         corpus_type="images" 使用）。
     返回统计（总数 / 成功识别 / 空或跳过）。

环境变量 EDR_RAG_OCR=0 可显式关闭 OCR（测试/离线环境加速）。缺依赖或
无法识别时优雅降级：ocr_text 记为 None/""，检索链路不受影响。
"""

from __future__ import annotations

import dataclasses
import logging
import os
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: 判定为「图片资产」的后缀（可用于 OCR 的点阵图/位图格式）。
#: .svg 是矢量图，tesseract 无法直接识别，故不含在其中。
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"}

#: OCR 显式关闭开关。
_ENV_DISABLE = os.environ.get("EDR_RAG_OCR", "1").strip().lower() in {"0", "false", "no", "off"}


@dataclasses.dataclass
class OcrSummary:
    """一次 run_ocr 的统计结果。"""
    total_images: int = 0
    ocr_succeeded: int = 0
    empty_or_skipped: int = 0
    engine: Optional[str] = None
    ocr_jsonl: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---------- OCR 引擎探测（可选依赖，缺依赖优雅降级） ----------

def _detect_pytesseract() -> Optional[Callable[[Path, str], str]]:
    """探测 pytesseract + 系统 tesseract，可用则返回 OCR 函数。"""
    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
    except Exception:  # pragma: no cover - 缺可选依赖
        return None

    try:
        # 确认系统里有 tesseract 二进制；缺失时 pytesseract 调用会抛错。
        # pytesseract.get_tesseract_version 找不到二进制会抛 TesseractNotFoundError。
        pytesseract.get_tesseract_version()
    except Exception:  # pragma: no cover - 系统 tesseract 未安装
        logger.debug("pytesseract available but no tesseract binary; OCR degraded")
        return None

    def _ocr(path: Path, lang_hint: str) -> str:
        try:
            with Image.open(path) as img:
                text = pytesseract.image_to_string(img, lang=lang_hint or "eng")
            return (text or "").strip()
        except Exception:  # pragma: no cover - 单图失败不阻断
            logger.debug("pytesseract OCR failed for %s", path, exc_info=True)
            return ""

    return _ocr


def _detect_rapidocr() -> Optional[Callable[[Path, str], str]]:
    """探测 rapidocr-onnxruntime（纯 Python，中英文内建），可用则返回 OCR 函数。"""
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:  # pragma: no cover - 缺可选依赖
        return None

    try:
        engine = RapidOCR()
    except Exception:  # pragma: no cover - 模型加载失败
        logger.debug("rapidocr engine init failed; OCR degraded", exc_info=True)
        return None

    def _ocr(path: Path, lang_hint: str) -> str:
        try:
            result, _ = engine(str(path))
        except Exception:  # pragma: no cover - 单图失败不阻断
            logger.debug("rapidocr OCR failed for %s", path, exc_info=True)
            return ""
        if not result:
            return ""
        lines = []
        for item in result:
            # RapidOCR 每项是 [box, text, score]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(str(item[1]))
        return " ".join(l for l in lines if l).strip()

    return _ocr


#: 已探测的引擎缓存（进程内只探测一次）。元素为 (OCR函数, 引擎名)；无后端时函数为 None。
_ENGINE_CACHE: Optional[tuple] = None


def get_ocr_engine() -> Optional[Callable[[Path, str], str]]:
    """返回可用的 OCR 函数；无可用后端时返回 None（主链路不报错）。

    返回函数签名为 fn(image_path: Path, lang_hint: str) -> str。
    """
    global _ENGINE_CACHE
    if _ENGINE_CACHE is not None:
        return _ENGINE_CACHE[0]
    if _ENV_DISABLE:
        _ENGINE_CACHE = (None, "disabled")
    else:
        f = _detect_pytesseract()
        if f is not None:
            _ENGINE_CACHE = (f, "pytesseract")
        else:
            f2 = _detect_rapidocr()
            _ENGINE_CACHE = (f2, "rapidocr" if f2 is not None else "-")
    if _ENGINE_CACHE[0] is None:
        logger.info("OCR engine unavailable; image OCR will be skipped (install 'ocr' extras)")
    return _ENGINE_CACHE[0]


def ocr_available() -> bool:
    """当前环境是否有可用的 OCR 后端。"""
    return get_ocr_engine() is not None


def _detect_lang(path: Path) -> Optional[str]:
    """基于图片内容/文件名尝试推测语言；无法判断返回 None。

    CHM 手册多为中文，若文件名含明显中文则倾向 chi_sim；否则交给
    tesseract 默认（调用方兜底。"chi_sim+eng" 常见于中英混排，这里
    直接给出稳妥的 lang 组合。
    """
    name = path.name
    if any("\u4e00" <= c <= "\u9fff" for c in name):
        return "chi_sim+eng"
    return None


def ocr_image(image_path: Path, lang: Optional[str] = None) -> str:
    """对单张图片做 OCR，返回识别文本。

    失败或无可识别内容始终返回空字符串（绝不上抛）。lang 为 None 时按
    _detect_lang 推测，仍无则退回 "eng"。
    """
    path = Path(image_path)
    if not path.is_file():
        return ""
    fn = get_ocr_engine()
    if fn is None:
        return ""
    lang_hint = lang or _detect_lang(path) or "eng"
    return fn(path, lang_hint)


# ---------- 批量流水线 ----------

def _is_image_asset(rec: dict) -> bool:
    if rec.get("type") != "image":
        return False
    p = Path(rec.get("path") or "")
    return p.suffix.lower() in IMAGE_SUFFIXES


def run_ocr(
    workdir: Path,
    *,
    lang: Optional[str] = None,
    force: bool = False,
) -> OcrSummary:
    """对 <workdir>/assets.jsonl 中的图片资产执行 OCR，产出可检索文本。

    行为：
      - 读 assets.jsonl；无该文件或非图片资产 → 直接返回空统计。
      - 无 OCR 引擎（缺依赖 / 被 EDR_RAG_OCR 关闭）→ 跳过，不写稿。
      - 对每个图片记录：已存在非空 ocr_text 且非 force → 复用（幂等，
        避免重复识别）；否则识别并写入。
      - 更新 assets.jsonl（每条图片记录写回 ocr_text 字段）。
      - 写出 <workdir>/ocr.jsonl：每条含 id/type/title/text/source，
        供 RetrievalService corpus_type="images" 检索。
      - 原子写 assets.jsonl 与 ocr.jsonl。

    force=True 时忽略已有 ocr_text 强制重识别（用于 OCR 引擎升级后重建）。
    """
    workdir = Path(workdir)
    assets_path = workdir / "assets.jsonl"
    if not assets_path.exists():
        logger.warning("No assets.jsonl at %s — run P0 ingest first", assets_path)
        return OcrSummary()

    records: list[dict] = []
    with assets_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(__import__("json").loads(line))
                except Exception:  # pragma: no cover - 防御坏行
                    logger.warning("skip unparseable asset line in %s", assets_path)

    fn = get_ocr_engine()
    summary = OcrSummary(engine=("pytesseract" if fn is not None else None))

    image_indices: list[int] = []
    for i, rec in enumerate(records):
        if _is_image_asset(rec):
            image_indices.append(i)
    summary.total_images = len(image_indices)

    if fn is None:
        # 缺依赖：直接按未识别写出 ocr.jsonl（空文本），主链路不报错。
        summary.empty_or_skipped = summary.total_images
        _write_ocr_jsonl(workdir, records, image_indices, summary)
        return summary

    for i in image_indices:
        rec = records[i]
        existing = rec.get("ocr_text")
        if existing and not force:
            # 幂等：已有文本则不重复识别（rename/更新场景）。
            if (existing or "").strip():
                summary.ocr_succeeded += 1
                continue
        full_path = Path(rec.get("absolute_path") or "") or (workdir / "extracted" / rec.get("path", ""))
        text = ocr_image(full_path, lang=lang)
        if text:
            rec["ocr_text"] = text
            summary.ocr_succeeded += 1
        else:
            rec["ocr_text"] = rec.get("ocr_text") or None  # 保留旧的或置 None
            summary.empty_or_skipped += 1

    _write_assets(workdir, records)
    _write_ocr_jsonl(workdir, records, image_indices, summary)
    return summary


def _write_assets(workdir: Path, records: list[dict]) -> None:
    import json

    from edr_rag.atomic_io import atomic_write_text

    body = "".join(
        json.dumps(r, ensure_ascii=False) + "\n"
        for r in records
    )
    atomic_write_text(workdir / "assets.jsonl", body)


def _write_ocr_jsonl(
    workdir: Path,
    records: list[dict],
    image_indices: list[int],
    summary: OcrSummary,
) -> None:
    import json

    from edr_rag.atomic_io import atomic_write_text

    out: list[dict] = []
    for i in image_indices:
        rec = records[i]
        text = (rec.get("ocr_text") or "").strip()
        out.append({
            "id": rec.get("id", ""),
            "type": "image",
            "title": "图片截图: " + rec.get("path", ""),
            "text": text,
            "source": {"file": rec.get("path", "")},
            "manual_id": rec.get("manual_id", ""),
            # 附带 alt/referenced_by，供上游展示上下文。
            "alt": rec.get("alt", ""),
            "referenced_by": rec.get("referenced_by", []),
            "ocr_text": text,  # 与 assets.jsonl 对齐，便于下游直接取
        })
    body = "".join(
        json.dumps(r, ensure_ascii=False) + "\n"
        for r in out
    )
    ocr_path = workdir / "ocr.jsonl"
    atomic_write_text(ocr_path, body)
    summary.ocr_jsonl = str(ocr_path)
