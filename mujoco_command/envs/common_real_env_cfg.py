from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CameraConfig:
    name: str
    type: str
    serial: str = ""


def _resolve_calib_dir(calib_dir: str) -> Path:
    raw_path = Path(calib_dir)
    if raw_path.is_absolute():
        return raw_path

    candidates = [
        Path.cwd() / raw_path,
        Path(__file__).resolve().parents[1] / raw_path,
        Path.cwd() / "envs" / "cfgs" / raw_path,
        Path(__file__).resolve().parent / "cfgs" / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _extract_intrinsics(payload: dict[str, Any]) -> np.ndarray:
    if "intrinsic_matrix" in payload:
        return np.asarray(payload["intrinsic_matrix"], dtype=float)
    if "matrix" in payload:
        return np.asarray(payload["matrix"], dtype=float)
    raise KeyError("Missing intrinsic_matrix/matrix in intrinsics payload")


def _extract_extrinsics(payload: dict[str, Any]) -> np.ndarray:
    if "transform_camera_to_robot" in payload:
        return np.asarray(payload["transform_camera_to_robot"], dtype=float)
    if "extrinsics" in payload:
        return np.asarray(payload["extrinsics"], dtype=float)
    raise KeyError("Missing transform_camera_to_robot/extrinsics in extrinsics payload")


@dataclass
class RealEnvConfig:
    wbc: int
    data_folder: str
    cameras: list[CameraConfig]
    pcl_cameras: list[str]
    calib_dir: str
    min_bound: list[float] = field(default_factory=list)
    max_bound: list[float] = field(default_factory=list)
    is_sim: int = 0
    intrinsics: dict[str, np.ndarray] = field(init=False, default_factory=dict)
    extrinsics: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.cameras = [
            camera if isinstance(camera, CameraConfig) else CameraConfig(**camera)
            for camera in self.cameras
        ]
        self._load_calibration_files()

    def _load_calibration_files(self):
        calib_root = _resolve_calib_dir(self.calib_dir)
        self.calib_dir = str(calib_root)

        for camera_name in self.pcl_cameras:
            intr_path = _pick_existing(
                [
                    calib_root / f"{camera_name}_intrinsics_real.json",
                    calib_root / f"{camera_name}_intrinsics_sim_fallback.json",
                ]
            )
            extr_path = _pick_existing(
                [
                    calib_root / f"{camera_name}_extrinsics_real.json",
                    calib_root / f"{camera_name}_extrinsics_model.json",
                ]
            )

            if intr_path is None:
                raise FileNotFoundError(
                    f"Missing intrinsics for {camera_name} under {calib_root}. "
                    f"Expected {camera_name}_intrinsics_real.json or "
                    f"{camera_name}_intrinsics_sim_fallback.json."
                )
            if extr_path is None:
                raise FileNotFoundError(
                    f"Missing extrinsics for {camera_name} under {calib_root}. "
                    f"Expected {camera_name}_extrinsics_real.json or "
                    f"{camera_name}_extrinsics_model.json."
                )

            self.intrinsics[camera_name] = _extract_intrinsics(_load_json(intr_path))
            self.extrinsics[camera_name] = _extract_extrinsics(_load_json(extr_path))
