from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SDK_ROOT = Path(os.environ.get("DEX5_SDK_ROOT", PROJECT_ROOT / "external/dex5_sdk"))
DEFAULT_SOURCE_URDF = SDK_ROOT / "Dex5-URDF-R/Dex5-URDF-R.urdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "sim/assets/dex5_urdf_import"

DEX5_JOINT_ORDER = [
    "Roll_21R",
    "Pitch_22R",
    "Pitch_23R",
    "Pitch_24R",
    "Roll_31R",
    "Pitch_32R",
    "Pitch_33R",
    "Pitch_34R",
    "Roll_41R",
    "Pitch_42R",
    "Pitch_43R",
    "Pitch_44R",
    "Roll_51R",
    "Pitch_52R",
    "Pitch_53R",
    "Pitch_54R",
    "Yaw_11R",
    "Roll_12R",
    "Pitch_13R",
    "Pitch_14R",
]

TIP_BODY_NAMES = ["Link_24R", "Link_34R", "Link_44R", "Link_54R", "Link_14R"]


def sanitize_urdf(source_urdf: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = source_urdf.parent
    output_urdf = output_dir / source_urdf.name

    mesh_src = source_dir / "meshes"
    mesh_dst = output_dir / "meshes"
    if mesh_dst.exists():
        shutil.rmtree(mesh_dst)
    shutil.copytree(mesh_src, mesh_dst)

    tree = ET.parse(source_urdf)
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename.startswith("meshes/"):
            mesh.attrib["filename"] = filename.removeprefix("meshes/")

    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    return output_urdf


def compile_and_save_mjcf(sanitized_urdf: Path, output_dir: Path) -> Path:
    model = mujoco.MjModel.from_xml_path(str(sanitized_urdf))
    output_mjcf = output_dir / "Dex5-URDF-R.compiled.xml"
    mujoco.mj_saveLastXML(str(output_mjcf), model)
    return output_mjcf


def _find_body(root: ET.Element, body_name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.attrib.get("name") == body_name:
            return body
    return None


def _disable_hand_mesh_collisions(root: ET.Element) -> None:
    for geom in root.iter("geom"):
        if geom.attrib.get("type") == "mesh":
            geom.attrib["contype"] = "0"
            geom.attrib["conaffinity"] = "0"


def _add_fingertip_proxy_geoms(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    existing_materials = {m.attrib.get("name") for m in asset.findall("material")}
    if "tip_proxy" not in existing_materials:
        ET.SubElement(asset, "material", {"name": "tip_proxy", "rgba": "0.10 0.85 0.45 0.45"})

    proxy_specs = {
        "Link_24R": ("0.020 -0.004 0.000", "0.012"),
        "Link_34R": ("0.020 -0.004 0.000", "0.012"),
        "Link_44R": ("0.020 -0.004 0.000", "0.012"),
        "Link_54R": ("0.018 -0.004 0.000", "0.011"),
        "Link_14R": ("0.026 0.002 0.000", "0.012"),
    }
    for body_name, (pos, size) in proxy_specs.items():
        body = _find_body(root, body_name)
        if body is None:
            continue
        geom_name = f"{body_name}_tip_proxy"
        if body.find(f"geom[@name='{geom_name}']") is not None:
            continue
        ET.SubElement(
            body,
            "geom",
            {
                "name": geom_name,
                "type": "sphere",
                "pos": pos,
                "size": size,
                "material": "tip_proxy",
                "friction": "2.0 0.08 0.01",
                "solref": "0.004 1",
                "solimp": "0.95 0.99 0.001",
            },
        )


def augment_mjcf_for_penspin(
    compiled_mjcf: Path,
    output_dir: Path,
    proxy_collision: bool = False,
    disable_mesh_collision: bool = False,
) -> Path:
    tree = ET.parse(compiled_mjcf)
    root = tree.getroot()
    root.attrib["model"] = "Dex5-PenSpin-RealURDF-M2-Proxy" if proxy_collision else "Dex5-PenSpin-RealURDF-M2"

    option = root.find("option")
    if option is None:
        compiler = root.find("compiler")
        option = ET.Element("option")
        root.insert(list(root).index(compiler) + 1 if compiler is not None else 0, option)
    option.attrib.update(
        {
            "timestep": "0.002",
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
        }
    )

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    material_specs = {
        "pen": "0.95 0.20 0.14 1",
        "marker_a": "0.1 0.3 1.0 1",
        "marker_b": "1.0 0.9 0.05 1",
        "floor": "0.45 0.45 0.45 1",
    }
    existing_materials = {m.attrib.get("name") for m in asset.findall("material")}
    for name, rgba in material_specs.items():
        if name not in existing_materials:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("compiled MJCF has no worldbody")
    if worldbody.find("light[@name='key']") is None:
        worldbody.insert(
            0,
            ET.Element(
                "light",
                {
                    "name": "key",
                    "pos": "0 -0.6 0.8",
                    "dir": "0 0.6 -0.8",
                    "directional": "true",
                },
            ),
        )
    if worldbody.find("camera[@name='overview']") is None:
        worldbody.insert(
            1,
            ET.Element(
                "camera",
                {
                    "name": "overview",
                    "pos": "0.22 -0.42 0.28",
                    "xyaxes": "0.88 0.48 0 -0.25 0.46 0.85",
                },
            ),
        )
    if worldbody.find("geom[@name='floor']") is None:
        worldbody.insert(
            2,
            ET.Element(
                "geom",
                {
                    "name": "floor",
                    "type": "plane",
                    "pos": "0 0 -0.07",
                    "size": "0.5 0.5 0.02",
                    "material": "floor",
                },
            ),
        )

    if worldbody.find("body[@name='pen']") is None:
        pen = ET.SubElement(worldbody, "body", {"name": "pen", "pos": "-0.02 0.035 0.085", "quat": "1 0 0 0"})
        ET.SubElement(pen, "freejoint", {"name": "pen_free"})
        ET.SubElement(
            pen,
            "geom",
            {
                "name": "pen_body",
                "type": "cylinder",
                "size": "0.006 0.085",
                "euler": "0 1.57079632679 0",
                "density": "450",
                "material": "pen",
                "friction": "2.0 0.08 0.01" if proxy_collision else "1.2 0.02 0.002",
                "solref": "0.004 1",
                "solimp": "0.95 0.99 0.001",
            },
        )
        ET.SubElement(
            pen,
            "geom",
            {
                "name": "pen_marker_a",
                "type": "sphere",
                "pos": "-0.083 0 0.006",
                "size": "0.007",
                "density": "50",
                "material": "marker_a",
            },
        )
        ET.SubElement(
            pen,
            "geom",
            {
                "name": "pen_marker_b",
                "type": "sphere",
                "pos": "0.083 0 -0.006",
                "size": "0.007",
                "density": "50",
                "material": "marker_b",
            },
        )
        ET.SubElement(pen, "site", {"name": "pen_end_a", "pos": "-0.085 0 0", "size": "0.004", "rgba": "0.1 0.3 1.0 1"})
        ET.SubElement(pen, "site", {"name": "pen_end_b", "pos": "0.085 0 0", "size": "0.004", "rgba": "1.0 0.9 0.05 1"})

    if proxy_collision and disable_mesh_collision:
        _disable_hand_mesh_collisions(root)
    if proxy_collision:
        _add_fingertip_proxy_geoms(root)

    joint_ranges: dict[str, str] = {}
    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        joint.attrib.setdefault("limited", "true")
        joint.attrib.setdefault("damping", "0.04")
        joint.attrib.setdefault("armature", "0.002")
        if "range" in joint.attrib:
            joint_ranges[name] = joint.attrib["range"]

    actuator = root.find("actuator")
    if actuator is not None:
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    for joint_name in DEX5_JOINT_ORDER:
        ctrlrange = joint_ranges[joint_name]
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"act_{joint_name}",
                "joint": joint_name,
                "kp": "8.0",
                "kv": "0.2",
                "ctrlrange": ctrlrange,
            },
        )

    if proxy_collision and disable_mesh_collision:
        name = "dex5_penspin_real_m2_proxy_only.xml"
    elif proxy_collision:
        name = "dex5_penspin_real_m2_mesh_plus_proxy.xml"
    else:
        name = "dex5_penspin_real_m2.xml"
    output_mjcf = output_dir / name
    ET.indent(tree, space="  ")
    tree.write(output_mjcf, encoding="utf-8", xml_declaration=False)
    return output_mjcf


def summarize_model(model_path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        for i in range(model.njnt)
    ]
    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        for i in range(model.nu)
    ]
    return {
        "path": str(model_path),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "njnt": int(model.njnt),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
        "joint_names": joint_names,
        "actuator_names": actuator_names,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-urdf", type=Path, default=DEFAULT_SOURCE_URDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    sanitized_urdf = sanitize_urdf(args.source_urdf, args.output_dir)
    compiled_mjcf = compile_and_save_mjcf(sanitized_urdf, args.output_dir)
    augmented_mjcf = augment_mjcf_for_penspin(compiled_mjcf, args.output_dir, proxy_collision=False)
    proxy_mjcf = augment_mjcf_for_penspin(compiled_mjcf, args.output_dir, proxy_collision=True, disable_mesh_collision=False)
    proxy_only_mjcf = augment_mjcf_for_penspin(compiled_mjcf, args.output_dir, proxy_collision=True, disable_mesh_collision=True)
    summary = summarize_model(sanitized_urdf)
    augmented_summary = summarize_model(augmented_mjcf)
    proxy_summary = summarize_model(proxy_mjcf)
    proxy_only_summary = summarize_model(proxy_only_mjcf)

    print("sanitized_urdf:", sanitized_urdf)
    print("compiled_mjcf:", compiled_mjcf)
    print("augmented_mjcf:", augmented_mjcf)
    print("proxy_mjcf:", proxy_mjcf)
    print("proxy_only_mjcf:", proxy_only_mjcf)
    print(
        "model:",
        f"nq={summary['nq']}",
        f"nv={summary['nv']}",
        f"nu={summary['nu']}",
        f"nbody={summary['nbody']}",
        f"njnt={summary['njnt']}",
        f"ngeom={summary['ngeom']}",
        f"nmesh={summary['nmesh']}",
    )
    print("joints:", ", ".join(summary["joint_names"]))
    print(
        "augmented:",
        f"nq={augmented_summary['nq']}",
        f"nv={augmented_summary['nv']}",
        f"nu={augmented_summary['nu']}",
        f"nbody={augmented_summary['nbody']}",
        f"njnt={augmented_summary['njnt']}",
        f"ngeom={augmented_summary['ngeom']}",
        f"nmesh={augmented_summary['nmesh']}",
    )
    print(
        "proxy:",
        f"nq={proxy_summary['nq']}",
        f"nv={proxy_summary['nv']}",
        f"nu={proxy_summary['nu']}",
        f"nbody={proxy_summary['nbody']}",
        f"njnt={proxy_summary['njnt']}",
        f"ngeom={proxy_summary['ngeom']}",
        f"nmesh={proxy_summary['nmesh']}",
    )
    print(
        "proxy_only:",
        f"nq={proxy_only_summary['nq']}",
        f"nv={proxy_only_summary['nv']}",
        f"nu={proxy_only_summary['nu']}",
        f"nbody={proxy_only_summary['nbody']}",
        f"njnt={proxy_only_summary['njnt']}",
        f"ngeom={proxy_only_summary['ngeom']}",
        f"nmesh={proxy_only_summary['nmesh']}",
    )


if __name__ == "__main__":
    main()
