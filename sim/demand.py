"""M2-W2b 需求曲线订单流 —— data/demand/ 曲线 → 连续订单发生器。

输入单一来源:
- data/demand/demand_profile.json: 每 SKU {base_rate_per_day, weekday_factor,
  month_factor, noise_cv} (W1 需求拟合产物);
- data/demand/sku_catalog.json: 每 SKU {abc, price, weight_class, weight_kg, ...}。

到达模型:
- 泊松到达, λ(t) = Σ_sku rate_sku(t), rate_sku(t) = base_rate_per_day
  × weekday_factor[weekday] × month_factor[month] × global_multiplier;
  global_multiplier 是全局可调倍率 (压力测试/降速演示用)。
- 非齐次泊松按 60s 桶数值积分: 每桶 Poisson(λ·60), 到达时刻桶内均匀;
  同 seed 严格可复现 (numpy PCG64)。
- noise_cv 是**日总量**噪声 (拟合残差), 不施加在到达时刻上; 需要
  日间波动时打开 daily_noise=True (每 SKU 每日乘 (1+N(0,cv)), 种子受控)。

订单语义:
- 每订单 = 1 SKU × qty (默认 1 件; max_qty>1 时 qty~U[1,max_qty]);
- OUTBOUND 出库单: 有该 SKU 库存的货位 → 同组交付台 D_G<r><c>;
- REPLENISH 补货单: 库存低于 replenish_threshold 时立即生成, 入库区
  INBOUND → 空闲货位。库存是「计划库存」: 出库开单即占用货位、补货开单
  即落位; 物理库存以调度层 COMMITTED 事务为准, 由上层对账
  (services/scheduler 七段语义)。384 货位 ≫ 40 SKU, 空闲货位不会枯竭。
- 24h continuous: generate_until(t) 可持续推进 cursor, 订单永续。

货位 id 解析借用 W1 的 parse_slot (懒加载, 布局单一来源不复制)。
本模块不依赖 mujoco 的任何计算 (仅货位 id 派生时惰性导入)。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE_PATH = REPO_ROOT / "data" / "demand" / "demand_profile.json"
DEFAULT_CATALOG_PATH = REPO_ROOT / "data" / "demand" / "sku_catalog.json"

#: 入库区节点名 (调度层坐标源, 见 sim/dispatch.py INBOUND_POS 打桩常量)
INBOUND_NODE = "INBOUND"

_SECONDS_PER_DAY = 86400.0
_BUCKET_S = 60.0          # NHPP 数值积分桶宽
_SEED_PALLETS = 48        # 与 LAYOUT["seed_pallets"] 对齐的初始在库托盘数


class OrderKind(str, Enum):
    OUTBOUND = "OUTBOUND"      # 货位 → 交付台
    REPLENISH = "REPLENISH"    # 入库区 → 货位


@dataclass(frozen=True)
class SkuInfo:
    """sku_catalog.json 一行 (重量/等级等调度常量的 SKU 侧来源)。"""

    sku: str
    abc: str
    price: float
    demand_mean_day: float
    cv: float
    weight_class: str
    weight_kg: float


@dataclass(frozen=True)
class DemandOrder:
    """一张订单: 调度层 build_task 的输入。"""

    order_id: str
    kind: OrderKind
    sku: str
    qty: int
    t_s: float                  # 仿真时钟 (s, 引擎起点为 0)
    src: str                    # OUTBOUND: 货位 / REPLENISH: INBOUND_NODE
    dst: str                    # OUTBOUND: D_G<r><c> / REPLENISH: 货位


def load_catalog(path: str | Path = DEFAULT_CATALOG_PATH) -> dict[str, SkuInfo]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        row["sku"]: SkuInfo(
            sku=row["sku"], abc=row["abc"], price=float(row["price"]),
            demand_mean_day=float(row["demand_mean_day"]), cv=float(row["cv"]),
            weight_class=row["weight_class"], weight_kg=float(row["weight_kg"]),
        )
        for row in raw
    }


def _slot_group_dock(slot_id: str) -> str:
    """货位 → 同组交付台 (R23-… → D_G23); 布局派生走 W1 单一来源。"""
    from sim.warehouse_cell import parse_slot      # 惰性: 避免 demand 硬依赖 mujoco

    r, c, _k, _l = parse_slot(slot_id)
    return f"D_G{r}{c}"


def _default_slots() -> list[str]:
    from sim.warehouse_cell import MatrixLayout

    return MatrixLayout().slot_ids()


class DemandEngine:
    """需求曲线 → 订单流。同 seed 同参数 ⇒ 逐单可复现。"""

    def __init__(
        self,
        *,
        profile_path: str | Path = DEFAULT_PROFILE_PATH,
        catalog_path: str | Path = DEFAULT_CATALOG_PATH,
        seed: int = 0,
        start_weekday: int = 0,       # 0=周一 … 6=周日 (profile 键同约定)
        start_month: int = 1,
        global_multiplier: float = 1.0,
        daily_noise: bool = False,
        slots: list[str] | None = None,
        initial_pallets: int = _SEED_PALLETS,
        replenish_threshold: int = 1,
        max_qty: int = 1,
    ):
        if global_multiplier <= 0:
            raise ValueError("global_multiplier 必须为正")
        if initial_pallets < self._catalog_count(catalog_path):
            raise ValueError("initial_pallets 少于 SKU 数, 无法保底每 SKU ≥1 托")
        self.profile: dict = json.loads(Path(profile_path).read_text(encoding="utf-8"))
        self.catalog: dict[str, SkuInfo] = load_catalog(catalog_path)
        missing = set(self.profile) - set(self.catalog)
        if missing:
            raise KeyError(f"profile 有而 catalog 无 SKU: {sorted(missing)[:5]}")
        self.skus: list[str] = sorted(self.profile)
        self.start_weekday = start_weekday
        self.start_month = start_month
        self.global_multiplier = float(global_multiplier)
        self.daily_noise = daily_noise
        self.replenish_threshold = int(replenish_threshold)
        self.max_qty = int(max_qty)

        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self.cursor_s = 0.0
        self._noise_days: dict[int, dict[str, float]] = {}
        self._n_out = 0
        self._n_rep = 0
        self.orders: list[DemandOrder] = []

        # ---- 计划库存: slot -> sku | None ----
        self.slots: list[str] = list(slots) if slots is not None else _default_slots()
        self._slot_sku: dict[str, str | None] = {s: None for s in self.slots}
        self._free: list[str] = list(self.slots)
        self._seed_inventory(initial_pallets)

    # ------------------------------------------------------------ 目录
    @staticmethod
    def _catalog_count(catalog_path) -> int:
        return len(json.loads(Path(catalog_path).read_text(encoding="utf-8")))

    def _seed_inventory(self, n_pallets: int) -> None:
        """初始在库: 每 SKU 保底 1 托 (出库流不被零库存截断), 余量按权重铺。"""
        pos = list(self._rng.permutation(len(self.slots)))
        pick = pos[:n_pallets]
        weights = np.array([self.profile[s]["base_rate_per_day"] for s in self.skus])
        weights = weights / weights.sum()
        for i, sku in enumerate(self.skus):
            self._place(self.slots[pick[i]], sku)          # 每 SKU 保底 1 托
        for idx in pick[len(self.skus):]:                  # 余量按权重铺
            sku = self.skus[int(self._rng.choice(len(self.skus), p=weights))]
            self._place(self.slots[idx], sku)

    def _place(self, slot: str, sku: str | None) -> None:
        if self._slot_sku.get(slot) is None and sku is not None:
            self._free.remove(slot)
        elif self._slot_sku.get(slot) is not None and sku is None:
            self._free.append(slot)
        self._slot_sku[slot] = sku

    # ------------------------------------------------------------ 速率
    def _weekday_of(self, t_s: float) -> int:
        return (self.start_weekday + int(t_s // _SECONDS_PER_DAY)) % 7

    def _sku_rate_per_day(self, sku: str, t_s: float) -> float:
        p = self.profile[sku]
        wd = p.get("weekday_factor", {})
        wf = float(wd[str(self._weekday_of(t_s))]) if wd else 1.0
        mf = float(p.get("month_factor", {}).get(str(self.start_month), 1.0))
        mult = self.global_multiplier * wf * mf
        if self.daily_noise:
            mult *= self._roll_day_noise(t_s)[sku]
        return float(p["base_rate_per_day"]) * mult

    def _roll_day_noise(self, t_s: float) -> dict[str, float]:
        day = int(t_s // _SECONDS_PER_DAY)
        if day not in self._noise_days:
            self._noise_days[day] = {
                s: 1.0 + float(self._rng.normal(
                    0.0, float(self.profile[s].get("noise_cv", 0.0))))
                for s in self.skus
            }
        return self._noise_days[day]

    def sku_rate_per_day(self, sku: str, t_s: float = 0.0) -> float:
        return self._sku_rate_per_day(sku, t_s)

    def rate_per_day(self, t_s: float = 0.0) -> float:
        """λ(t): 全场订单速率 (单/天)。"""
        return sum(self._sku_rate_per_day(s, t_s) for s in self.skus)

    def expected_orders(self, t0_s: float = 0.0, t1_s: float | None = None) -> float:
        """期望订单数 = ∫λ dt (桶积分, 与生成器同口径)。"""
        t1 = self.cursor_s if t1_s is None else t1_s
        total = 0.0
        t = t0_s
        while t < t1 - 1e-9:
            dt = min(_BUCKET_S, t1 - t)
            total += self.rate_per_day(t + dt / 2) / _SECONDS_PER_DAY * dt
            t += dt
        return total

    def expected_sku_orders(self, t0_s: float = 0.0,
                            t1_s: float | None = None) -> dict[str, float]:
        t1 = self.cursor_s if t1_s is None else t1_s
        out = {s: 0.0 for s in self.skus}
        t = t0_s
        while t < t1 - 1e-9:
            dt = min(_BUCKET_S, t1 - t)
            for s in self.skus:
                out[s] += self._sku_rate_per_day(s, t + dt / 2) / _SECONDS_PER_DAY * dt
            t += dt
        return out

    # ------------------------------------------------------------ 库存视图
    def stock_by_sku(self, sku: str) -> int:
        return sum(1 for v in self._slot_sku.values() if v == sku)

    def inventory(self) -> dict[str, str | None]:
        return dict(self._slot_sku)

    def free_slot_count(self) -> int:
        return len(self._free)

    # ------------------------------------------------------------ 生成
    def generate_until(self, t_end_s: float) -> list[DemandOrder]:
        """推进 cursor 到 t_end_s, 返回新产生的订单 (含触发的补货单)。"""
        if t_end_s < self.cursor_s:
            raise ValueError("t_end_s 早于 cursor, 时间不能倒流")
        fresh: list[DemandOrder] = []
        t = self.cursor_s
        while t < t_end_s - 1e-9:
            dt = min(_BUCKET_S, t_end_s - t)
            lam = self.rate_per_day(t + dt / 2) / _SECONDS_PER_DAY * dt
            n = int(self._rng.poisson(lam))
            if n:
                offs = np.sort(self._rng.random(n)) * dt
                for off in offs:
                    fresh.extend(self._emit(t + off))
            t += dt
        self.cursor_s = t_end_s
        self.orders.extend(fresh)
        return fresh

    def stream(self, t_end_s: float):
        """逐单惰性产出 (24h continuous 用), 内部按桶推进。"""
        while self.cursor_s < t_end_s:
            nxt = min(self.cursor_s + _BUCKET_S, t_end_s)
            for o in self.generate_until(nxt):
                yield o

    def _emit(self, t_s: float) -> list[DemandOrder]:
        """一次到达: 按 rate 权重采样 SKU; 零库存则丢弃 (保底铺货后基本不触发)。"""
        rates = np.array([self._sku_rate_per_day(s, t_s) for s in self.skus])
        if rates.sum() <= 0:
            return []
        sku = self.skus[int(self._rng.choice(len(self.skus), p=rates / rates.sum()))]
        stocked = [s for s, v in self._slot_sku.items() if v == sku]
        if not stocked:
            return []
        qty = (int(self._rng.integers(1, self.max_qty + 1))
               if self.max_qty > 1 else 1)
        slot = stocked[int(self._rng.integers(len(stocked)))]
        self._n_out += 1
        out = DemandOrder(
            order_id=f"ORD-{self._n_out:06d}", kind=OrderKind.OUTBOUND,
            sku=sku, qty=qty, t_s=round(t_s, 3), src=slot,
            dst=_slot_group_dock(slot),
        )
        # 出库开单即占用 → 计划库存 −1; 低于阈值立即补货开单 (落位即 +1)
        self._place(slot, None)
        if self.stock_by_sku(sku) < self.replenish_threshold:
            return [out, self._emit_replenish(t_s, sku)]
        return [out]

    def _emit_replenish(self, t_s: float, sku: str) -> DemandOrder:
        if not self._free:
            raise RuntimeError("无空闲货位可补货 (384 货位不应枯竭)")
        slot = self._free[int(self._rng.integers(len(self._free)))]
        self._n_rep += 1
        order = DemandOrder(
            order_id=f"REP-{self._n_rep:06d}", kind=OrderKind.REPLENISH,
            sku=sku, qty=1, t_s=round(t_s, 3), src=INBOUND_NODE, dst=slot,
        )
        self._place(slot, sku)          # 补货开单即计划落位
        return order


def one_hour_sanity(engine: DemandEngine) -> dict:
    """生成 1h 订单流并给出与 profile 对比的统计 (订单数 / SKU 分布)。"""
    t0 = engine.cursor_s
    fresh = engine.generate_until(t0 + 3600.0)
    out = [o for o in fresh if o.kind is OrderKind.OUTBOUND]
    rep = [o for o in fresh if o.kind is OrderKind.REPLENISH]
    exp_total = engine.expected_orders(t0, t0 + 3600.0)
    exp_sku = engine.expected_sku_orders(t0, t0 + 3600.0)
    got_sku: dict[str, int] = {s: 0 for s in engine.skus}
    for o in out:
        got_sku[o.sku] += 1
    return {
        "window_s": (t0, t0 + 3600.0),
        "orders": len(out), "replenish": len(rep),
        "expected": exp_total,
        "rel_err_total": (abs(len(out) - exp_total) / max(exp_total, 1e-9)),
        "sku_counts": got_sku,
        "sku_expected": {k: v for k, v in exp_sku.items() if v > 0},
    }
