"""图像扫码：渲染帧 → 解码 → 格式/CRC 校验 → 注册表 → 实体 ID。

流程对应设计文档 21.3：检测码 → 解码文本 → 格式与 CRC 检查 → 查注册表；
任务目标核对由调用方（任务/调度层）完成。

硬规则：本模块只消费图像帧与冻结注册表（标注/身份映射用），
不读取任何场景对象属性；物体隐藏属性只进测试答案通道。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import zxingcpp

from services.perception.labels import LabelRegistry, TokenFormatError, parse_token, verify_crc

QR_SYMBOL_MODULES = 21  # 不含静区；zxing position 覆盖 21×21 符号区


class ScanFailureReason(str, Enum):
    NO_SYMBOL = "no_symbol"          # 帧内未检测到码（含静区遮挡等）
    DECODE_ERROR = "decode_error"    # 检出轮廓但无法解出文本（失焦/像素不足等）
    BAD_FORMAT = "bad_format"        # 文本不符合 8 字节令牌格式
    CRC_MISMATCH = "crc_mismatch"    # 格式正确但 CRC 校验失败
    UNKNOWN_TOKEN = "unknown_token"  # CRC 正确但注册表中无此令牌


@dataclass(frozen=True)
class ScanFailure:
    reason: ScanFailureReason
    detail: str = ""
    frame_sha256: str = ""


@dataclass(frozen=True)
class ScanResult:
    """一次扫码的完整结果：成功给实体 ID，失败给 ScanFailure。

    frame_id 由调用方携带（三次尝试必须不同 frame_id），
    frame_sha256 是原图内容引用，用于拒绝重复帧/旧帧复用。
    """

    frame_id: str
    frame_sha256: str
    token: str | None
    entity_id: str | None
    failure: ScanFailure | None
    px_per_module: float | None  # 解码器实测投影模块边长（像素）， None=未解码

    @property
    def ok(self) -> bool:
        return self.failure is None and self.entity_id is not None


def frame_sha256(image_bgr: np.ndarray) -> str:
    """原图内容哈希（BGR 字节流 sha256），作图像引用。"""
    arr = np.ascontiguousarray(image_bgr)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def px_per_module_from_position(position) -> float:
    """由 zxingcpp 返回的符号四角坐标估计投影后最短模块边长（像素）。

    按第 21 章口径取透视后最短投影边：min(宽, 高)/21。
    """
    corners = np.array(
        [
            [position.top_left.x, position.top_left.y],
            [position.top_right.x, position.top_right.y],
            [position.bottom_left.x, position.bottom_left.y],
        ],
        dtype=np.float64,
    )
    width = float(np.linalg.norm(corners[1] - corners[0]))
    height = float(np.linalg.norm(corners[2] - corners[0]))
    return min(width, height) / QR_SYMBOL_MODULES


def _decode_texts(image_bgr: np.ndarray) -> tuple[list[str], float | None]:
    """解码候选文本。优先 zxing-cpp，退回 cv2.QRCodeDetector。

    返回 (候选文本列表, px_per_module 或 None)。px_per_module 仅 zxing 可得。
    """
    try:
        results = zxingcpp.read_barcodes(image_bgr)
    except Exception:
        results = []
    texts: list[str] = []
    px_mod: float | None = None
    for res in results:
        if res.format == zxingcpp.BarcodeFormat.QRCode and res.text:
            texts.append(res.text)
            try:
                px_mod = px_per_module_from_position(res.position)
            except Exception:
                px_mod = None
            break  # 感知只处理首个 QR；多码帧由调用方裁剪视场
    if texts:
        return texts, px_mod
    # cv2 退路（无位置信息，px_per_module 保持 None）
    detector = cv2.QRCodeDetector()
    text, points, _ = detector.detectAndDecode(image_bgr)
    if text:
        return [text], None
    if points is not None:
        return [], None  # 检出轮廓但解不出文本
    return [], None


def scan_image(
    image_bgr: np.ndarray,
    registry: LabelRegistry,
    frame_id: str = "",
) -> ScanResult:
    """扫描一帧 BGR 图像，返回实体 ID 或 ScanFailure。"""
    sha = frame_sha256(image_bgr)

    def fail(reason: ScanFailureReason, detail: str = "") -> ScanResult:
        return ScanResult(
            frame_id=frame_id,
            frame_sha256=sha,
            token=None,
            entity_id=None,
            failure=ScanFailure(reason=reason, detail=detail, frame_sha256=sha),
            px_per_module=None,
        )

    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.ndim < 2:
        return fail(ScanFailureReason.NO_SYMBOL, "无效图像帧")

    texts, px_mod = _decode_texts(image_bgr)
    if not texts:
        return fail(ScanFailureReason.NO_SYMBOL)

    token = texts[0]
    try:
        parse_token(token)
    except TokenFormatError as exc:
        return fail(ScanFailureReason.BAD_FORMAT, str(exc))
    if not verify_crc(token):
        return fail(ScanFailureReason.CRC_MISMATCH, f"{token} CRC8/0x07 校验失败")
    entity_id = registry.entity_for_token(token)
    if entity_id is None:
        return fail(ScanFailureReason.UNKNOWN_TOKEN, f"{token} 不在冻结注册表")
    return ScanResult(
        frame_id=frame_id,
        frame_sha256=sha,
        token=token,
        entity_id=entity_id,
        failure=None,
        px_per_module=px_mod,
    )
