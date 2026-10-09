"""W04 货位标签 — 把 18 个货位绑定到冻结标签注册表, 并提供渲染-解码回路。

设计约束 (21.2 / 25.4):
* 感知**只吃渲染帧**, 不读世界真值判定身份; 真值仅用于误差评测。
* 标签映射冻结: 令牌↔货位 id 在运行期不可变, 冲突时拒绝构建。
* 图像质量与定位健康门槛拒绝不可信输入, 不用真值放行。

本模块负责"哪个货位贴哪枚标签"以及端到端自检 (渲染→解码→比对),
不做任务决策。
"""
from __future__ import annotations

from dataclasses import dataclass

from services.perception.labels import TOKEN_KINDS, LabelRegistry, LabelSpec
from services.perception.render import render_label_frame
from services.perception.scan import ScanResult, scan_image

# 货位标签类别: 令牌类型位是**单字符** (design 21.2 的 TOKEN_KINDS="PSLG"),
# "S" = slot (货位)。P=托盘 / L=层 / G=组 是同一位宽的其他类别。
SLOT_KIND = "S"
assert SLOT_KIND in TOKEN_KINDS, f"标签类别 {SLOT_KIND} 不在 {TOKEN_KINDS} 中"


@dataclass(frozen=True)
class SlotLabelBinding:
    """一个货位的标签绑定。"""

    slot_id: str
    number: int

    @property
    def spec(self) -> LabelSpec:
        return LabelSpec(kind=SLOT_KIND, number=self.number, entity_id=self.slot_id)


def build_slot_registry(slot_ids: list[str], start_number: int = 1) -> LabelRegistry:
    """按货位顺序生成冻结注册表；编号连续且不与既有令牌冲突。"""
    bindings = [SlotLabelBinding(slot_id, start_number + i)
                for i, slot_id in enumerate(slot_ids)]
    return LabelRegistry(b.spec for b in bindings)


def bindings_of(registry: LabelRegistry) -> list[SlotLabelBinding]:
    """从注册表还原 (货位, 编号) 绑定列表。"""
    out = []
    for spec in registry.specs:
        if spec.kind != SLOT_KIND:
            continue
        out.append(SlotLabelBinding(spec.entity_id, spec.number))
    return sorted(out, key=lambda b: b.number)


class SlotLabelRenderer:
    """按货位标签渲染可被解码的图像帧。

    view_deg 用于斜视预算核对 (设计 21.2 要求 45° 仍可解)。
    """

    def __init__(self, registry: LabelRegistry, *, view_deg: float = 0.0,
                 distance_m: float = 0.150):
        self.registry = registry
        self.view_deg = float(view_deg)
        self.distance_m = float(distance_m)

    def render(self, slot_id: str) -> bytes:
        """返回该货位标签的 PNG 纹理字节。"""
        token = self.registry.token_for_entity(slot_id)
        if token is None:
            raise KeyError(f"货位 {slot_id} 未注册标签")
        from services.perception.labels import render_qr_png
        return render_qr_png(token)

    def capture(self, slot_id: str) -> ScanResult:
        """渲染→解码一帧, 返回扫描结果（含令牌、实体、失败原因）。"""
        frame = render_label_frame(
            self.render(slot_id), view_deg=self.view_deg,
            distance_m=self.distance_m)
        return scan_image(frame, self.registry, frame_id=f"{slot_id}@{self.view_deg:.0f}deg")

    def self_check(self, slot_ids: list[str] | None = None) -> dict[str, ScanResult]:
        """对给定货位（默认全部）跑一次渲染-解码回路。"""
        ids = slot_ids if slot_ids is not None else [sp.entity_id for sp in self.registry.specs()]
        return {sid: self.capture(sid) for sid in ids}