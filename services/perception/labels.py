"""标签令牌、注册表与 QR 纹理生成。

规格来源：设计文档第 21 章（docs/桌面AGV与机器人仓储Demo_阶段一仿真详细设计_v2.0.0.md）
与 configs/design.yaml ``labels:`` 段。

令牌格式（8 字节）::

    [类型1字符][编号4位]['|'][CRC 2位大写hex]
    类型: P=托盘 S=SKU L=货位 G=货物
    CRC8: 多项式 0x07、初值 0x00、无位反转、无输出异或，
          覆盖 '|' 之前的 5 个 ASCII 字符。

QR: Version 1、纠错等级 M、四周 4 模块静区，含静区每边 29 模块。
编码器必须校验未升级版本且模块数正确，否则拒绝导出。

标签纹理只以图像方式贴到几何表面；注册表是标注/答案比对用的真值，
不得进入感知输入（见 AGENTS.md 硬规则 1）。
"""

from __future__ import annotations

import argparse
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

import qrcode
from qrcode.exceptions import DataOverflowError

# ---------------------------------------------------------------------------
# 常量（与 configs/design.yaml labels: 段一致）
# ---------------------------------------------------------------------------

TOKEN_KINDS = "PSLG"
TOKEN_SEPARATOR = "|"
TOKEN_LENGTH = 8
TOKEN_RE = re.compile(r"^([PSLG])(\d{4})\|([0-9A-F]{2})$")

CRC8_POLY = 0x07
CRC8_INIT = 0x00

QR_VERSION = 1
QR_ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_M
QUIET_ZONE_MODULES = 4
QR_MODULES = 21  # Version 1
MODULES_PER_SIDE = QR_MODULES + 2 * QUIET_ZONE_MODULES  # 29，含静区
TOKEN_BYTE_CAPACITY_V1M = 14  # QR V1-M byte mode，令牌 8 字节可容纳

# 编号位数与取值范围
NUMBER_DIGITS = 4
NUMBER_MAX = 10**NUMBER_DIGITS - 1


# ---------------------------------------------------------------------------
# CRC8
# ---------------------------------------------------------------------------


def crc8(data: bytes, poly: int = CRC8_POLY, init: int = CRC8_INIT) -> int:
    """CRC-8：多项式 0x07、初值 0、MSB first（无位反转）、输出异或 0。

    标准校验向量：crc8(b"123456789") == 0xF4（CRC catalog "CRC-8"）。
    """
    crc = init & 0xFF
    for byte in data:
        crc ^= byte & 0xFF
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


# ---------------------------------------------------------------------------
# 令牌
# ---------------------------------------------------------------------------


class TokenFormatError(ValueError):
    """令牌格式、类型或编号非法。"""


def make_token(kind: str, number: int) -> str:
    """由类型与编号生成 8 字节令牌，如 make_token("P", 1) == "P0001|DC"。"""
    if kind not in TOKEN_KINDS:
        raise TokenFormatError(f"非法令牌类型 {kind!r}，必须是 {TOKEN_KINDS} 之一")
    if not 0 <= number <= NUMBER_MAX:
        raise TokenFormatError(f"编号 {number} 超出 0..{NUMBER_MAX}")
    payload = f"{kind}{number:0{NUMBER_DIGITS}d}"
    return f"{payload}{TOKEN_SEPARATOR}{crc8(payload.encode('ascii')):02X}"


def parse_token(token: str) -> tuple[str, int, str]:
    """解析令牌，返回 (类型, 编号, CRC hex)。格式非法抛 TokenFormatError。"""
    m = TOKEN_RE.match(token)
    if m is None:
        raise TokenFormatError(f"令牌 {token!r} 不符合 {TOKEN_LENGTH} 字节格式")
    return m.group(1), int(m.group(2)), m.group(3)


def verify_crc(token: str) -> bool:
    """校验令牌尾部 CRC；格式非法返回 False。"""
    try:
        _, number, crc_hex = parse_token(token)
    except TokenFormatError:
        return False
    payload = f"{token[0]}{number:04d}"
    return crc8(payload.encode("ascii")) == int(crc_hex, 16)


# ---------------------------------------------------------------------------
# 注册表：令牌 ↔ 实体 ID（生成纹理前冻结）
# ---------------------------------------------------------------------------


class RegistryConflictError(ValueError):
    """令牌或实体 ID 冲突，禁止导出纹理。"""


@dataclass(frozen=True)
class LabelSpec:
    """一枚标签的注册条目。"""

    kind: str
    number: int
    entity_id: str

    @property
    def token(self) -> str:
        return make_token(self.kind, self.number)


class LabelRegistry:
    """冻结的 令牌↔实体ID 映射。

    构建时检查冲突：同一令牌映射多个实体、或同一实体持有多个令牌都拒绝；
    构建后内部映射不可变（只读视图）。
    """

    def __init__(self, entries: Iterable[LabelSpec]):
        token_to_entity: dict[str, str] = {}
        entity_to_token: dict[str, str] = {}
        meta: dict[str, LabelSpec] = {}
        for entry in entries:
            token = entry.token
            if token in token_to_entity:
                raise RegistryConflictError(
                    f"令牌 {token} 冲突: {token_to_entity[token]!r} 与 {entry.entity_id!r}"
                )
            if entry.entity_id in entity_to_token:
                raise RegistryConflictError(
                    f"实体 {entry.entity_id!r} 同时持有令牌 "
                    f"{entity_to_token[entry.entity_id]} 与 {token}"
                )
            token_to_entity[token] = entry.entity_id
            entity_to_token[entry.entity_id] = token
            meta[token] = entry
        self._token_to_entity = token_to_entity
        self._entity_to_token = entity_to_token
        self._meta = meta

    # -- 查询 ---------------------------------------------------------------

    @property
    def tokens(self) -> Mapping[str, str]:
        """只读 {token: entity_id} 视图。"""
        return MappingProxyType(self._token_to_entity)

    def entity_for_token(self, token: str) -> str | None:
        """已知令牌返回实体 ID，未知令牌返回 None。"""
        return self._token_to_entity.get(token)

    def token_for_entity(self, entity_id: str) -> str | None:
        return self._entity_to_token.get(entity_id)

    def __contains__(self, token: object) -> bool:
        return token in self._token_to_entity

    def __len__(self) -> int:
        return len(self._token_to_entity)

    def specs(self) -> tuple[LabelSpec, ...]:
        return tuple(self._meta.values())

    # -- 序列化 --------------------------------------------------------------

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "format": "desktop-agv-demo/label-registry/1",
            "crc8": {"poly": CRC8_POLY, "init": CRC8_INIT, "xorout": 0x00, "reflect": False},
            "qr": {"version": QR_VERSION, "ecc": "M", "quiet_zone_modules": QUIET_ZONE_MODULES},
            "labels": [
                {"token": s.token, "kind": s.kind, "number": s.number, "entity_id": s.entity_id}
                for s in sorted(self._meta.values(), key=lambda s: (s.kind, s.number))
            ],
        }
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "LabelRegistry":
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            LabelSpec(kind=e["kind"], number=int(e["number"]), entity_id=e["entity_id"])
            for e in doc["labels"]
        )


def default_entity_id(kind: str, number: int) -> str:
    """缺省实体 ID 命名：P→PAL-nnn、S→SKU-nnn、G→G-nnn、L→A-Ln-Cn。"""
    if kind == "P":
        return f"PAL-{number:03d}"
    if kind == "S":
        return f"SKU-{number:03d}"
    if kind == "G":
        return f"G-{number:03d}"
    if kind == "L":
        layer = number // 10
        column = number % 10
        return f"A-L{layer}-C{column}"
    raise TokenFormatError(f"非法令牌类型 {kind!r}")


# ---------------------------------------------------------------------------
# QR 纹理 PNG
# ---------------------------------------------------------------------------


class QrVersionError(ValueError):
    """编码器升级了 QR 版本或模块数不符，按规格禁止导出。"""


def qr_matrix(text: str) -> list[list[bool]]:
    """编码文本为 Version 1 (21×21) 模块矩阵；溢出或版本漂移抛 QrVersionError。"""
    qr = qrcode.QRCode(
        version=QR_VERSION,
        error_correction=QR_ERROR_CORRECTION,
        box_size=1,
        border=QUIET_ZONE_MODULES,
    )
    qr.add_data(text)
    try:
        qr.make(fit=False)
    except DataOverflowError as exc:
        raise QrVersionError(f"文本 {text!r} 超出 QR V{QR_VERSION}-M 容量: {exc}") from exc
    if qr.version != QR_VERSION or qr.modules_count != QR_MODULES:
        raise QrVersionError(
            f"QR 版本漂移: 期望 V{QR_VERSION} ({QR_MODULES}×{QR_MODULES})，"
            f"实际 V{qr.version} ({qr.modules_count}×{qr.modules_count})"
        )
    return qr.get_matrix()


def render_qr_png(text: str, box_size: int = 8) -> bytes:
    """生成含静区的 QR PNG 字节流。

    box_size 是每模块的纹素数；V1+4 静区 → 29*box_size 见方。
    先经 qr_matrix() 强制 V1 校验，再按模块数核对输出像素尺寸。
    """
    qr_matrix(text)  # 版本/容量校验（升级即抛 QrVersionError）
    img = qrcode.QRCode(
        version=QR_VERSION,
        error_correction=QR_ERROR_CORRECTION,
        box_size=box_size,
        border=QUIET_ZONE_MODULES,
    )
    img.add_data(text)
    img.make(fit=False)
    pil = img.make_image(fill_color="black", back_color="white").convert("RGB")
    expected = MODULES_PER_SIDE * box_size
    if pil.size != (expected, expected):
        raise QrVersionError(f"PNG 尺寸 {expected}x{expected} 预期不符，实际 {pil.size}")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def write_qr_png(text: str, path: str | Path, box_size: int = 8) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_qr_png(text, box_size=box_size))
    return path


# ---------------------------------------------------------------------------
# 批量生成入口
# ---------------------------------------------------------------------------

_DEFAULT_COUNTS = {"P": 4, "S": 3, "G": 8, "L": 12}


def generate_label_set(
    out_dir: str | Path,
    counts: Mapping[str, int] | None = None,
    box_size: int = 8,
) -> LabelRegistry:
    """生成默认标签集：registry.json + 每令牌一张 PNG，返回冻结注册表。

    覆盖同名文件（幂等）；冲突在 LabelRegistry 构建时即拒绝。
    """
    out_dir = Path(out_dir)
    counts = dict(_DEFAULT_COUNTS if counts is None else counts)
    entries: list[LabelSpec] = []
    for kind in TOKEN_KINDS:
        for number in range(counts.get(kind, 0)):
            entries.append(
                LabelSpec(kind=kind, number=number, entity_id=default_entity_id(kind, number))
            )
    registry = LabelRegistry(entries)
    registry.to_json(out_dir / "registry.json")
    for spec in registry.specs():
        write_qr_png(spec.token, out_dir / f"{spec.token.replace('|', '_')}.png", box_size=box_size)
    return registry


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="生成标签令牌注册表与 QR 纹理 PNG")
    ap.add_argument("--out", default="assets/labels", help="产物目录（默认 assets/labels）")
    for kind in TOKEN_KINDS:
        ap.add_argument(
            f"--{kind.lower()}",
            type=int,
            default=_DEFAULT_COUNTS[kind],
            help=f"{kind} 类标签数量（默认 %d）" % _DEFAULT_COUNTS[kind],
        )
    ap.add_argument("--box-size", type=int, default=8, help="每模块纹素数（默认 8）")
    args = ap.parse_args(argv)
    counts = {k: getattr(args, k.lower()) for k in TOKEN_KINDS}
    registry = generate_label_set(args.out, counts=counts, box_size=args.box_size)
    print(f"{len(registry)} labels -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
