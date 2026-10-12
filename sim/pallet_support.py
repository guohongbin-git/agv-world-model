"""M2W2 — 库存台账 (mission/demo 层; 物理真值来自 WarehouseCell 播种)。

位置编码: "slot:R11-S0" / "stand:H02" / "dock:D_G12" / "carried:vid"。
SKU 标签由调度层 (dispatch) 挂载 —— 本台账只跟踪托盘物理位置。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from sim.warehouse_cell import pallet_name_for


@dataclass
class PalletLedger:
    loc: dict = field(default_factory=dict)          # pallet → 位置
    sku_of: dict = field(default_factory=dict)       # pallet → SKU id (可选)

    def at(self, where: str) -> str | None:
        for p, w in self.loc.items():
            if w == where:
                return p
        return None

    def move(self, pallet: str, dest: str):
        assert self.loc.get(pallet) is not None, f"{pallet} 不在台账中"
        self.loc[pallet] = dest

    def snapshot(self) -> dict:
        return dict(sorted(self.loc.items()))


def ledger_from_cell(cell) -> PalletLedger:
    led = PalletLedger()
    for name in cell.pallets:
        if name.startswith("pallet_R"):
            parts = name.split("_")                   # pallet R11 S0 L2
            led.loc[name] = f"slot:{parts[1]}-S{parts[2][1:]}"
        elif name.startswith("pallet_H"):
            led.loc[name] = f"stand:{name[len('pallet_'):]}"
        elif name.startswith("pallet_D"):
            led.loc[name] = f"dock:{name[len('pallet_'):]}"
    return led
