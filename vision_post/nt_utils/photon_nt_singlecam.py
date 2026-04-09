# photon_nt_singlecam.py
from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import ntcore

from config.camera_config import CameraConfig
from config.app_config import AppConfig
from geometry_utils.distance_utils import distance_from_pitch
from geometry_utils.pose_utils import camera_pose2d_calculate
from geometry_utils.ballpose_utils import ball_xy_from_camera
from nt_utils.photon_decode import decode_pipeline_result
from nt_utils.pose2d_reader import Pose2d


@dataclass
class CameraState:
    seq: Optional[int] = None
    targets: List[Dict[str, Any]] = field(default_factory=list)
    multitag_present: int = 0
    leftover: int = 0
    raw_len: int = 0
    last_error: Optional[str] = None
    last_update_monotonic: float = 0.0


@dataclass
class SingleCamBallObservation:
    camera_name: str
    target_index: int
    yaw_deg: float
    pitch_deg: float
    area: float
    confidence: Optional[float]
    distance_m: float
    ball_x: float
    ball_y: float
    raw_target: Dict[str, Any]


class PhotonSingleCamClient:
    """
    Simplified PhotonVision NT client for a single camera.

    - 連 NT server
    - 訂閱 /photonvision/<CameraName>/rawBytes
    - 保留最新一筆解析後的 targets
    - 支援屬性讀取：<CameraName>_Yaw / <CameraName>_Pitch / <CameraName>_BestYaw / <CameraName>_Targets
    """

    def __init__(
        self,
        server: str,
        camera_name: str,
        client_name: str = "orangepi-singlecam",
        table_name: str = "photonvision",
        poll_dt: float = 0.02,
        sort_targets_by_area_desc: bool = True,
    ):
        self.server = server
        self.camera_name = camera_name
        self.client_name = client_name
        self.table_name = table_name
        self.poll_dt = poll_dt
        self.sort_targets_by_area_desc = sort_targets_by_area_desc

        self._inst = ntcore.NetworkTableInstance.create()
        self._sub: Optional[Any] = None
        self._state = CameraState()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def start(self) -> None:
        self._inst.startClient4(self.client_name)
        self._inst.setServer(self.server)

        root = self._inst.getTable(self.table_name)
        subtable = root.getSubTable(self.camera_name)
        self._sub = subtable.getRawTopic("rawBytes").subscribe("raw", b"")

        self._thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()

    def _cam_loop(self) -> None:
        if self._sub is None:
            return

        last_seq = None

        while not self._stop_evt.is_set():
            raw = self._sub.get()
            if not raw:
                time.sleep(self.poll_dt)
                continue

            try:
                md, targets, mt_present, leftover = decode_pipeline_result(raw)
                seq = md.get("sequenceID", None)

                if seq is not None and seq == last_seq:
                    time.sleep(self.poll_dt)
                    continue
                last_seq = seq

                if self.sort_targets_by_area_desc and targets:
                    targets = sorted(targets, key=lambda t: float(t.get("area", 0.0)), reverse=True)

                with self._lock:
                    self._state.seq = seq
                    self._state.targets = list(targets)
                    self._state.multitag_present = mt_present
                    self._state.leftover = leftover
                    self._state.raw_len = len(raw)
                    self._state.last_error = None
                    self._state.last_update_monotonic = time.monotonic()

            except Exception as e:
                with self._lock:
                    self._state.last_error = f"{type(e).__name__}: {e}"

            time.sleep(self.poll_dt)

    def get_state(self) -> CameraState:
        with self._lock:
            st = self._state
            return CameraState(
                seq=st.seq,
                targets=list(st.targets),
                multitag_present=st.multitag_present,
                leftover=st.leftover,
                raw_len=st.raw_len,
                last_error=st.last_error,
                last_update_monotonic=st.last_update_monotonic,
            )

    def _get_field_list(self, key: str) -> List[Any]:
        with self._lock:
            return [t.get(key) for t in self._state.targets]

    def _get_best_field(self, key: str):
        with self._lock:
            if not self._state.targets:
                return None
            return self._state.targets[0].get(key)

    def __getattr__(self, name: str):
        if "_" not in name:
            raise AttributeError(name)

        cam, field = name.split("_", 1)
        if cam != self.camera_name:
            raise AttributeError(name)

        field_map = {
            "Yaw": "yaw",
            "Pitch": "pitch",
            "Area": "area",
            "Skew": "skew",
            "Conf": "objDetectConf",
            "ObjId": "objDetectId",
            "Fid": "fiducialId",
            "Amb": "poseAmbiguity",
        }

        if field == "Seq":
            with self._lock:
                return self._state.seq

        if field == "Targets":
            with self._lock:
                return list(self._state.targets)

        if field.startswith("Best"):
            base = field[len("Best") :]
            if base not in field_map:
                raise AttributeError(name)
            return self._get_best_field(field_map[base])

        if field in field_map:
            return self._get_field_list(field_map[field])

        raise AttributeError(name)

    def compute_ball_observations(
        self,
        robot_pose: Pose2d,
        cam_cfg: CameraConfig,
        app_cfg: AppConfig,
    ) -> List[SingleCamBallObservation]:
        """
        計算單台相機的球觀測結果。

        依序:
          - 取最新 targets
          - 用 pitch 計算距離
          - 用 yaw 投影成場地座標

        若資料不完整或計算失敗，會跳過該 target。
        """
        observations: List[SingleCamBallObservation] = []

        if robot_pose is None:
            return observations
        if cam_cfg.name != self.camera_name:
            return observations

        state = self.get_state()
        if state.last_error is not None:
            return observations

        targets = state.targets
        if not targets:
            return observations

        camera_pose = camera_pose2d_calculate(
            robot_pose=robot_pose,
            forward_m=cam_cfg.forward_m,
            left_m=cam_cfg.left_m,
            camera_yaw_offset_deg=cam_cfg.camera_yaw_offset_deg,
        )

        for i, target in enumerate(targets):
            yaw = target.get("yaw")
            pitch = target.get("pitch")
            if yaw is None or pitch is None:
                continue

            yaw_deg = float(yaw)
            pitch_deg = float(pitch)
            area = float(target.get("area", 0.0))
            conf_raw = target.get("objDetectConf")
            confidence = float(conf_raw) if conf_raw is not None else None

            distance_m = distance_from_pitch(
                pitch_deg=pitch_deg,
                camera_height_m=cam_cfg.height_m,
                camera_pitch_deg=cam_cfg.pitch_deg,
                target_height_m=app_cfg.target_height_m,
                eps=app_cfg.distance_eps,
            )
            if distance_m is None:
                continue
            if distance_m < cam_cfg.min_distance_m:
                continue
            if distance_m > cam_cfg.max_distance_m:
                continue

            ball_x, ball_y = ball_xy_from_camera(
                camera_pose=camera_pose,
                yaw_deg=yaw_deg,
                distance_m=distance_m,
                yaw_sign=cam_cfg.yaw_sign,
            )

            observations.append(
                SingleCamBallObservation(
                    camera_name=self.camera_name,
                    target_index=i,
                    yaw_deg=yaw_deg,
                    pitch_deg=pitch_deg,
                    area=area,
                    confidence=confidence,
                    distance_m=distance_m,
                    ball_x=ball_x,
                    ball_y=ball_y,
                    raw_target=target,
                )
            )

        return observations
