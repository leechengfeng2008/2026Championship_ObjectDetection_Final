from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import ntcore
from wpimath.geometry import Pose2d as WpiPose2d


@dataclass
class Pose2d:
    x: float
    y: float
    heading_rad: float


class Pose2dReader:
    """
    Read AdvantageKit Pose2d struct topic from NT4.

    修正版重點：
    - 不再直接把 subscribe 的預設 WpiPose2d() 當成有效資料
    - 使用 readQueue() 只處理真正從 NT 收到的新資料
    - 在尚未收到第一筆真資料前，get_pose2d() 會回傳 None
    - 收到資料後，若暫時沒有新更新，則回傳 last_good
    """

    def __init__(
        self,
        server: str,
        topic_path: str,
        client_name: str = "orangepi-pose2d-reader",
    ):
        self.server = server
        self.topic_path = topic_path

        self._inst = ntcore.NetworkTableInstance.create()
        self._inst.startClient4(client_name)
        self._inst.setServer(server)

        # 仍需提供 default，但後續不直接信任 get() 的結果
        self._sub = self._inst.getStructTopic(topic_path, WpiPose2d).subscribe(WpiPose2d())

        self._last_good: Optional[Pose2d] = None
        self._has_received_real_update = False
        self._last_update_monotonic: float = 0.0

    def _convert_raw_pose(self, raw: WpiPose2d) -> Pose2d:
        """
        將 wpimath Pose2d 轉成你專案內使用的簡化 Pose2d。
        """
        try:
            x = float(raw.x)
            y = float(raw.y)
        except Exception:
            # 某些版本可能需要用 X()/Y()
            x = float(raw.X())
            y = float(raw.Y())

        heading_rad = float(raw.rotation().radians())
        return Pose2d(x=x, y=y, heading_rad=heading_rad)

    def _drain_queue(self) -> None:
        """
        把 NT queue 中最新的 pose 全部吃掉，只保留最後一筆。
        若 queue 為空，代表沒有真正的新資料。
        """
        try:
            updates = self._sub.readQueue()
        except Exception:
            # 若環境/版本不支援 readQueue，就退回舊方式。
            # 但這種情況下無法完全避免 default pose 風險。
            raw = self._sub.get()
            if raw is None:
                return

            p = self._convert_raw_pose(raw)
            self._last_good = p
            self._has_received_real_update = True
            self._last_update_monotonic = time.monotonic()
            return

        if not updates:
            return

        latest_value = None

        for item in updates:
            # 不同版本的 ntcore queue item 可能欄位名不同，
            # 這裡做較保守的兼容處理。
            if hasattr(item, "value"):
                latest_value = item.value
            else:
                latest_value = item

        if latest_value is None:
            return

        p = self._convert_raw_pose(latest_value)
        self._last_good = p
        self._has_received_real_update = True
        self._last_update_monotonic = time.monotonic()

    def get_pose2d(self) -> Optional[Pose2d]:
        """
        取得目前最新 pose。

        行為：
        - 若 queue 有新資料，更新 last_good
        - 若已收過真資料但目前沒有新資料，回傳 last_good
        - 若從未收過真資料，回傳 None
        """
        self._drain_queue()

        if not self._has_received_real_update:
            return None

        return self._last_good

    def has_pose2d(self) -> bool:
        """
        是否已經收過至少一筆真實 pose 更新。
        """
        self._drain_queue()
        return self._has_received_real_update

    def get_last_update_age_s(self) -> Optional[float]:
        """
        距離最後一次收到真實 pose 更新，已過多久（秒）。
        若尚未收到任何真資料，回傳 None。
        """
        self._drain_queue()

        if not self._has_received_real_update:
            return None

        return time.monotonic() - self._last_update_monotonic