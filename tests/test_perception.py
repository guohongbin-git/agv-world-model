"""感知前置件验收：令牌/CRC、注册表、MuJoCo 渲染往返、像素预算、拒绝路径。

对应 T07/T08 前置（设计文档第 7.2/21 章）：
  1. CRC8/0x07 手算向量与目录校验值
  2. 注册表生成与冲突拒绝（冻结映射）
  3. 核心：MuJoCo 相机渲染 QR 纹理 → 图像解码回原令牌 → 实体 ID
  4. 45° 斜视仍可解码，且实测最短投影模块边 ≥4 px
  5. CRC 错误令牌、未知令牌、非法格式、无码帧被正确拒绝

硬规则对照：感知只吃渲染图像；真值注册表只作身份映射/答案比对，
测试不向 scan_image 传任何场景对象属性。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from services.perception import (
    MODULES_PER_SIDE,
    LabelRegistry,
    LabelSpec,
    RegistryConflictError,
    ScanFailureReason,
    TokenFormatError,
    QrVersionError,
    crc8,
    default_entity_id,
    frontal_px_per_module,
    make_token,
    parse_token,
    render_label_frame,
    render_qr_png,
    scan_image,
    verify_crc,
    write_qr_png,
)

MIN_PX_PER_MODULE = 4.0  # design.yaml: min_projected_px_per_module


def make_registry() -> LabelRegistry:
    return LabelRegistry(
        [
            LabelSpec("P", 1, "PAL-001"),
            LabelSpec("P", 2, "PAL-002"),
            LabelSpec("G", 1, "G-001"),
            LabelSpec("S", 1, "SKU-001"),
            LabelSpec("L", 32, "A-L3-C2"),
        ]
    )


# ---------------------------------------------------------------------------
# 1. CRC8 与令牌格式
# ---------------------------------------------------------------------------


class TestCrc8:
    def test_catalog_check_value(self):
        # CRC catalog "CRC-8" (poly 0x07, init 0, no reflect, xorout 0)
        assert crc8(b"123456789") == 0xF4

    def test_hand_computed_vector_P0001(self):
        # 逐位手算：'P'→0xB7, "P0"→0x9C, "P00"→0x4D, "P000"→0x74, "P0001"→0xDC
        assert crc8(b"P0001") == 0xDC
        assert make_token("P", 1) == "P0001|DC"

    def test_independent_table_implementation(self):
        # 表驱动（查表法）独立实现对拍，防实现相互复制
        def _crc8_table(data: bytes, poly: int = 0x07) -> int:
            table = []
            for i in range(256):
                v = i
                for _ in range(8):
                    v = ((v << 1) ^ poly) & 0xFF if v & 0x80 else (v << 1) & 0xFF
                table.append(v)
            crc = 0
            for b in data:
                crc = table[crc ^ b]
            return crc

        samples = [b"", b"P0000", b"P0001", b"S9999", b"L0032", b"G0100", b"abcde"]
        for s in samples:
            assert crc8(s) == _crc8_table(s)

    def test_token_format_and_crc_flag(self):
        token = make_token("G", 7)
        kind, number, crc_hex = parse_token(token)
        assert (kind, number) == ("G", 7) and len(token) == 8
        assert verify_crc(token)
        broken = token[:-2] + ("00" if token[-2:] != "00" else "FF")
        assert not verify_crc(broken)
        with pytest.raises(TokenFormatError):
            make_token("X", 1)  # 类型不在 PSLG
        with pytest.raises(TokenFormatError):
            make_token("P", 10000)  # 编号超 4 位
        with pytest.raises(TokenFormatError):
            parse_token("P001|AB")  # 缺一位编号
        assert not verify_crc("P0001|dc")  # 小写 hex 不合格式（须大写）

    def test_default_entity_ids(self):
        assert default_entity_id("P", 1) == "PAL-001"
        assert default_entity_id("S", 12) == "SKU-012"
        assert default_entity_id("G", 3) == "G-003"
        assert default_entity_id("L", 32) == "A-L3-C2"


# ---------------------------------------------------------------------------
# 2. 注册表：生成、冲突拒绝、冻结
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_build_and_lookup(self):
        reg = make_registry()
        assert len(reg) == 5
        assert reg.entity_for_token("P0001|DC") == "PAL-001"
        assert reg.entity_for_token("L0032|" + make_token("L", 32)[-2:]) == "A-L3-C2"
        assert reg.token_for_entity("G-001") == make_token("G", 1)
        assert "P0001|DC" in reg
        assert "P0001|FF" not in reg

    def test_duplicate_token_conflict(self):
        with pytest.raises(RegistryConflictError, match="P0001"):
            LabelRegistry(
                [LabelSpec("P", 1, "PAL-001"), LabelSpec("P", 1, "PAL-002")]
            )

    def test_duplicate_entity_conflict(self):
        with pytest.raises(RegistryConflictError, match="PAL-001"):
            LabelRegistry(
                [LabelSpec("P", 1, "PAL-001"), LabelSpec("G", 1, "PAL-001")]
            )

    def test_frozen_mapping_rejects_mutation(self):
        reg = make_registry()
        with pytest.raises(TypeError):
            reg.tokens["P0009|XX"] = "PAL-009"  # type: ignore[index]

    def test_json_roundtrip(self, tmp_path):
        reg = make_registry()
        path = reg.to_json(tmp_path / "registry.json")
        reg2 = LabelRegistry.from_json(path)
        assert len(reg2) == len(reg)
        for spec in reg.specs():
            assert reg2.entity_for_token(spec.token) == spec.entity_id


# ---------------------------------------------------------------------------
# 3. QR 纹理生成：V1/M、模块数、静区
# ---------------------------------------------------------------------------


class TestQrGeneration:
    def test_png_dimensions_and_decode(self, tmp_path):
        token = make_token("P", 1)
        path = write_qr_png(token, tmp_path / "P0001_DC.png")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert img.shape[0] == img.shape[1] == MODULES_PER_SIDE * 8  # 29 模块 × 8 纹素
        # 图像直读（不经渲染）先验纹理本身可解码
        res = scan_image(img, make_registry(), frame_id="texture")
        assert res.ok and res.entity_id == "PAL-001" and res.token == token

    def test_version_overflow_rejected(self):
        with pytest.raises(QrVersionError):
            render_qr_png("x" * 15)  # V1-M 字节容量 14，15 字节必升级 → 拒绝导出


# ---------------------------------------------------------------------------
# 4. 核心往返：MuJoCo 渲染 → 解码（正对 / 45° 斜视 / 像素预算）
# ---------------------------------------------------------------------------


class TestRenderRoundtrip:
    REG = make_registry()

    def test_frontal_roundtrip(self):
        token = make_token("P", 1)
        frame = render_label_frame(render_qr_png(token), view_deg=0.0)
        assert frame.dtype == np.uint8 and frame.shape == (960, 1280, 3)
        res = scan_image(frame, self.REG, frame_id="frontal-001")
        assert res.ok
        assert res.token == token and res.entity_id == "PAL-001"
        # 像素预算：实测（zxing 四角投影）且与理论一致
        assert res.px_per_module is not None
        assert res.px_per_module >= MIN_PX_PER_MODULE
        assert res.px_per_module == pytest.approx(frontal_px_per_module(), rel=0.15)

    def test_oblique_45_roundtrip_with_px_budget(self):
        token = make_token("G", 1)
        frame = render_label_frame(render_qr_png(token), view_deg=45.0)
        res = scan_image(frame, self.REG, frame_id="oblique-045")
        assert res.ok
        assert res.token == token and res.entity_id == "G-001"
        # 第 21 章口径：按透视后最短模块投影边核对，理论 1280×15×cos45°/100/29 ≈ 4.68 px
        assert res.px_per_module is not None
        assert res.px_per_module >= MIN_PX_PER_MODULE
        assert res.px_per_module == pytest.approx(
            frontal_px_per_module() * np.cos(np.pi / 4), rel=0.15
        )

    def test_each_registered_label_roundtrips(self):
        # 冻结注册表内每枚令牌都能经渲染-解码找回实体（P/G/S/L 四类）
        for spec in self.REG.specs():
            frame = render_label_frame(render_qr_png(spec.token), view_deg=20.0)
            res = scan_image(frame, self.REG, frame_id=f"sweep-{spec.token[:5]}")
            assert res.ok, f"{spec.entity_id} ({spec.token}) 渲染往返失败"
            assert res.entity_id == spec.entity_id


# ---------------------------------------------------------------------------
# 5. 拒绝路径（全部经真实渲染帧）
# ---------------------------------------------------------------------------


class TestRejections:
    REG = make_registry()

    def _scan_png(self, text: str, frame_id: str, view_deg: float = 30.0):
        frame = render_label_frame(render_qr_png(text), view_deg=view_deg)
        return scan_image(frame, self.REG, frame_id=frame_id)

    def test_crc_error_rejected(self):
        res = self._scan_png("P0001|FF", "rej-crc")
        assert not res.ok
        assert res.failure.reason == ScanFailureReason.CRC_MISMATCH
        assert res.entity_id is None

    def test_unknown_token_rejected(self):
        # 格式与 CRC 都正确，但不在冻结注册表
        token = make_token("P", 9999)
        res = self._scan_png(token, "rej-unknown")
        assert not res.ok
        assert res.failure.reason == ScanFailureReason.UNKNOWN_TOKEN
        assert res.failure.detail and "P9999" in res.failure.detail

    def test_bad_format_rejected(self):
        res = self._scan_png("PAL-001", "rej-format")  # 真实实体 ID 直接当码文本
        assert not res.ok
        assert res.failure.reason == ScanFailureReason.BAD_FORMAT

    def test_no_symbol_on_blank_frame(self):
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        res = scan_image(blank, self.REG, frame_id="blank")
        assert not res.ok
        assert res.failure.reason == ScanFailureReason.NO_SYMBOL

    def test_frame_hash_is_content_reference(self):
        t1 = make_token("P", 1)
        f1 = render_label_frame(render_qr_png(t1), view_deg=0.0)
        r1 = scan_image(f1, self.REG, frame_id="h-1")
        r1_again = scan_image(f1, self.REG, frame_id="h-1-again")  # 同图不同 frame_id
        assert r1.frame_sha256 == r1_again.frame_sha256 != ""
        other = render_label_frame(render_qr_png(make_token("P", 2)), view_deg=0.0)
        r2 = scan_image(other, self.REG, frame_id="h-2")
        assert r2.frame_sha256 != r1.frame_sha256  # 不同内容 → 不同图像引用
