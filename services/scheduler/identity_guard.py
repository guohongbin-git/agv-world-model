"""W05 身份核验闸门 — T08「错误拦截」与 T15「工位互斥」的前置。

设计判据:
|T08 错误拦截|错产品 错托盘 错货位各 20 次|60 次均不放行, 库存不错误提交|
|T15 工位互斥|M01/M02活动、收拢未达、收拢旧帧、R01未退出各10次|
         |进入请求全部拒绝; 合法STOW_CLEAR与版本一致后原子放行|

原则: **核验失败一律拒绝, 不得"尽量放行"**。三种身份都要核:
1. 货物 goods_id —— 抓取前确认手上这件确实是任务指定的;
2. 托盘 pallet_id —— 装/取前确认托盘身份;
3. 货位 slot_id  —— 入库前确认目标货位身份。

任一不符 → 拒绝 + 记录原因, 库存**零变化**。这是 T08 的核心:
拒绝不得留下半成品状态。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from services.perception.scan import ScanFailureReason, ScanResult


class RejectReason(str, Enum):
    WRONG_GOODS = "wrong_goods"        # 错产品
    WRONG_PALLET = "wrong_pallet"      # 错托盘
    WRONG_SLOT = "wrong_slot"          # 错货位
    NO_IDENTITY = "no_identity"        # 证据不足 (非 UNKNOWN 令牌)
    STALE_FRAME = "stale_frame"        # 旧帧 / 重复帧
    BUSY = "busy"                      # 他人在位 (T15)


@dataclass
class Verification:
    ok: bool
    reason: Optional[RejectReason] = None
    detail: str = ""
    scan: Optional[ScanResult] = None

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class IdentityGuard:
    """身份核验闸门 + 已用帧集合 (拒绝重复帧复用)。"""

    registry_entity_ids: frozenset[str] = frozenset()
    _seen_frames: set[str] = field(default_factory=set)

    # ------------------------------------------------------------ 扫码核验
    def verify_scan(self, result: ScanResult, expected: str,
                    *, allow_reuse: bool = False) -> Verification:
        """核验一次扫码结果是否指向期望实体。

        设计 R08 要求「独立试验ID、原帧、真值与解码输出」, 故重复帧
        默认拒绝 —— 同一张图不能当两次独立观测。
        """
        if not allow_reuse:
            if result.frame_sha256 in self._seen_frames:
                return Verification(False, RejectReason.STALE_FRAME,
                                    f"重复帧 {result.frame_sha256[:12]}", result)
            self._seen_frames.add(result.frame_sha256)
        if not result.ok:
            # 全部扫码失败原因 (无符号/解码错/格式错/CRC错/未知令牌) 都归为
            # 「无身份证据」—— 都不能放行, 但都不等同于「扫到了别的」,
            # 报告里要能区分是「看不清」还是「看错了」。
            return Verification(False, RejectReason.NO_IDENTITY,
                                f"扫码失败: {result.failure.reason.value} "
                                f"{result.failure.detail}", result)
        if result.entity_id != expected:
            return Verification(False, RejectReason.WRONG_GOODS,
                                f"扫到 {result.entity_id}, 期望 {expected}", result)
        return Verification(True, scan=result)

    # ------------------------------------------------------------ 三重核验
    def verify_handoff(self, *, expected_goods: str, expected_pallet: str,
                       expected_slot: Optional[str] = None,
                       scanned_goods: Optional[ScanResult] = None,
                       scanned_pallet: Optional[ScanResult] = None,
                       scanned_slot: Optional[ScanResult] = None) -> Verification:
        """装/取/放前的三重身份核验。任一不符立即拒绝。"""
        for scan, expected, why in ((scanned_goods, expected_goods, RejectReason.WRONG_GOODS),
                                    (scanned_pallet, expected_pallet, RejectReason.WRONG_PALLET),
                                    (scanned_slot, expected_slot, RejectReason.WRONG_SLOT)):
            if scan is None:
                if expected is None:
                    continue
                return Verification(False, RejectReason.NO_IDENTITY,
                                    f"缺 {why.value} 观测")
            if not scan.ok:
                return Verification(False, RejectReason.NO_IDENTITY,
                                    f"{why.value} 扫码失败: "
                                    f"{scan.failure.reason.value}", scan)
            if scan.entity_id != expected:
                return Verification(False, why,
                                    f"扫到 {scan.entity_id}, 期望 {expected}", scan)
        return Verification(True)

    def reset_frames(self) -> None:
        """新一轮试验清空帧集合 (T08 每类 20 次须是 20 次**独立**观测)。"""
        self._seen_frames.clear()