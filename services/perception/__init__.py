"""感知服务：标签生成、图像扫码、MuJoCo 相机渲染桥。

公开面：
  - labels: 令牌/CRC8、冻结注册表、QR 纹理 PNG 生成
  - scan:   渲染帧 → 解码 → CRC → 注册表 → 实体 ID（或 ScanFailure）
  - render: QR 纹理 → MuJoCo 相机渲染帧（感知输入的唯一来源）
"""

from services.perception.labels import (
    MODULES_PER_SIDE,
    QR_MODULES,
    QUIET_ZONE_MODULES,
    LabelRegistry,
    LabelSpec,
    RegistryConflictError,
    TokenFormatError,
    QrVersionError,
    crc8,
    default_entity_id,
    generate_label_set,
    make_token,
    parse_token,
    render_qr_png,
    verify_crc,
    write_qr_png,
)
from services.perception.render import frontal_px_per_module, render_label_frame
from services.perception.scan import (
    ScanFailure,
    ScanFailureReason,
    ScanResult,
    frame_sha256,
    px_per_module_from_position,
    scan_image,
)

__all__ = [
    "MODULES_PER_SIDE",
    "QR_MODULES",
    "QUIET_ZONE_MODULES",
    "LabelRegistry",
    "LabelSpec",
    "RegistryConflictError",
    "TokenFormatError",
    "QrVersionError",
    "crc8",
    "default_entity_id",
    "generate_label_set",
    "make_token",
    "parse_token",
    "render_qr_png",
    "verify_crc",
    "write_qr_png",
    "frontal_px_per_module",
    "render_label_frame",
    "ScanFailure",
    "ScanFailureReason",
    "ScanResult",
    "frame_sha256",
    "px_per_module_from_position",
    "scan_image",
]
