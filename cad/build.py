"""W02 参数化白模 — 导出入口。

运行: ``./.venv/bin/python -m cad.build``

产物（全部由 configs/design.yaml 参数化生成）:
* assets/cad/<model>.step   — 工程交换（内部米制模型, 导出边界一次转 mm 口径）;
* assets/cad/<model>.stl    — 视觉网格（米, MuJoCo/Blender 直接可用）;
* assets/collision/collision.yaml — box/cylinder 碰撞基元清单（模型局部系）;
* assets/cad/mjcf_skeleton.xml    — MuJoCo 骨架（与碰撞清单同源 BodyNode 树）;
* manifests/cad_manifest.json     — design_version/文件 sha256/输入参数哈希/
  导出程序版本/生成时间/派生参数审计（设计文档 8.1）。
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import build123d
import yaml
from build123d import Unit, export_step, export_stl

from .mjcf import build_mjcf
from .models import BodyNode, Prim, build_all_models
from .params import Derived, config_sha256, derive_parameters, load_params

REPO_ROOT = Path(__file__).resolve().parents[1]
CAD_DIR = "assets/cad"
COLLISION_PATH = "assets/collision/collision.yaml"
MJCF_NAME = "mjcf_skeleton.xml"
MANIFEST_PATH = "manifests/cad_manifest.json"

STL_TOLERANCE_M = 1e-4        # 0.1 mm 弦差（视觉网格）
STL_ANGULAR_TOL = 0.2


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prim_to_dict(prim: Prim) -> dict:
    out: dict = {"name": prim.name, "type": prim.type, "center_m": list(prim.center_m)}
    if prim.type == "box":
        out["size_m"] = list(prim.size_m)
    else:
        out.update(radius_m=prim.radius_m, height_m=prim.height_m, axis=prim.axis)
    if prim.friction is not None:
        out["friction"] = prim.friction
    return out


def _bodies_to_dicts(root: BodyNode) -> dict:
    """BodyNode 树 → 碰撞清单 bodies 字典（key 为 body 路径, a.b.c）。"""
    bodies: dict = {}

    def walk(node: BodyNode, path: str, parent: BodyNode | None) -> None:
        key = path or node.name
        entry: dict = {"pos_m": list(node.pos_m)}
        if node.quat_wxyz is not None:
            entry["quat_wxyz"] = list(node.quat_wxyz)
        if node.joint is not None:
            entry["joint"] = {"name": node.joint.name, "kind": node.joint.kind,
                              "axis": list(node.joint.axis)}
            if node.joint.range_rad_or_m is not None:
                entry["joint"]["range"] = list(node.joint.range_rad_or_m)
            if node.joint.drive_max is not None:
                entry["joint"]["drive_max"] = node.joint.drive_max
        if node.mass_kg is not None:
            entry["mass_kg"] = node.mass_kg
            if node.com_x_m is not None:
                entry["com_x_m"] = node.com_x_m
        if node.prims:
            entry["primitives"] = [_prim_to_dict(p) for p in node.prims]
        if node.sites:
            entry["sites"] = {k: list(v) for k, v in node.sites.items()}
        if node.note:
            entry["note"] = node.note
        bodies[key] = entry
        for child in node.children:
            walk(child, f"{key}.{child.name}", node)

    walk(root, "", None)
    return bodies


def _collision_doc(p, models: dict[str, BodyNode]) -> dict:
    return {
        "design_version": p.design_version,
        "units": {"length": "m", "mass": "kg"},
        "conventions": (
            "模型局部坐标系见 cad/models.py 模块文档; 原点约定与 MJCF body 帧一致。"
            "非凸体（托盘）已分解为多 box, 叉孔空隙保留; 圆柱轴向见 axis 字段。"
        ),
        "models": {
            name: {"origin_note": node.note, "bodies": _bodies_to_dicts(node)}
            for name, node in models.items()
        },
    }


def build_all(root: str | Path | None = None) -> dict:
    """构建全部模型并落盘产物; 返回 manifest 字典。"""
    root = Path(root).resolve() if root else REPO_ROOT
    cfg_path = root / "configs" / "design.yaml"
    p = load_params(cfg_path)
    derived: Derived = derive_parameters(p)
    models = build_all_models(p, derived)

    cad_dir = root / CAD_DIR
    collision_path = root / COLLISION_PATH
    manifest_path = root / MANIFEST_PATH
    for directory in (cad_dir, collision_path.parent, manifest_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    files: list[dict] = []
    model_infos: dict[str, dict] = {}

    for name, node in models.items():
        step_path = cad_dir / f"{name}.step"
        stl_path = cad_dir / f"{name}.stl"
        if not export_step(node.shape, step_path, unit=Unit.M):
            raise RuntimeError(f"STEP 导出失败: {name}")
        if not export_stl(node.shape, stl_path, tolerance=STL_TOLERANCE_M,
                          angular_tolerance=STL_ANGULAR_TOL):
            raise RuntimeError(f"STL 导出失败: {name}")
        for path, kind in ((step_path, "step"), (stl_path, "stl")):
            files.append({"path": str(path.relative_to(root)), "kind": kind,
                          "bytes": path.stat().st_size, "sha256": _sha256(path)})
        model_infos[name] = {
            "step": str(step_path.relative_to(root)),
            "stl": str(stl_path.relative_to(root)),
            "mass_kg": node.mass_kg,
            "bodies": len(_bodies_to_dicts(node)),
            "primitives": sum(len(n.prims) for n in node.iter_nodes()),
            "sites": sorted({f"{n.name}.{s}" if n is not node else s
                             for n in node.iter_nodes() for s in n.sites}),
            "note": node.note,
        }

    collision_doc = _collision_doc(p, models)
    with open(collision_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(collision_doc, f, allow_unicode=True, sort_keys=False,
                       default_flow_style=False)
    files.append({"path": COLLISION_PATH, "kind": "collision",
                  "bytes": collision_path.stat().st_size, "sha256": _sha256(collision_path)})

    mjcf_path = cad_dir / MJCF_NAME
    mjcf_path.write_text(build_mjcf(p, derived, models), encoding="utf-8")
    files.append({"path": str(mjcf_path.relative_to(root)), "kind": "mjcf",
                  "bytes": mjcf_path.stat().st_size, "sha256": _sha256(mjcf_path)})

    manifest = {
        "manifest_version": "1",
        "stage": "W02",
        "design_version": p.design_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "units": {"length": "m", "mass": "kg", "angle": "rad", "time": "s"},
        "export_units": {"step": "mm（由米制模型在导出边界一次转换）", "stl": "m"},
        "inputs": {"config": str(cfg_path.relative_to(root)),
                   "config_sha256": config_sha256(cfg_path)},
        "tools": {"generator": "cad.build", "python": platform.python_version(),
                  "build123d": build123d.__version__},
        "derived_parameters": {"values": derived.values, "notes": derived.notes},
        "models": model_infos,
        "files": files,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"cad.build: design_version={p.design_version} 模型 {len(models)} 个, "
          f"产物 {len(files)} 个文件")
    for name, info in model_infos.items():
        print(f"  - {name}: bodies={info['bodies']} prims={info['primitives']} "
              f"sites={len(info['sites'])} mass={info['mass_kg']}")
    print(f"  collision -> {COLLISION_PATH}")
    print(f"  mjcf      -> {CAD_DIR}/{MJCF_NAME}")
    print(f"  manifest  -> {MANIFEST_PATH}")
    return manifest


if __name__ == "__main__":
    sys.exit(0 if build_all() else 1)
