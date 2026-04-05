# photon_decode.py
from __future__ import annotations

import struct
from typing import Any, Dict, List, Tuple


#  corner -> 影像平面上的一個點 (x, y)
Corner = Tuple[float, float]


class Buf:
    """
    二進位讀取器。

    - self.b = 原始 bytes
    - self.i = 目前讀取位置（offset）
    - 每讀一次資料，就把 self.i 往後推

    依照 PhotonVision 的封包格式，從前到後逐段解包。
    """

    def __init__(self, b: bytes):
        self.b = b
        self.i = 0

    def remaining(self) -> int:
        """
        回傳目前還剩多少 bytes 尚未讀取。
        """
        return len(self.b) - self.i

    def _need(self, n: int):
        """
        在讀取 n bytes 之前，先確認 buffer 剩餘長度足夠。
        若不足，代表封包不完整或格式假設錯誤，直接丟出錯誤。
        """
        if self.remaining() < n:
            raise ValueError(
                f"buffer underrun: need {n}, remaining {self.remaining()}"
            )

    def u8(self) -> int:
        """
        讀取 1 byte 的 unsigned int。
        """
        self._need(1)
        v = self.b[self.i]
        self.i += 1
        return v

    def i32(self) -> int:
        """
        讀取 4 bytes little-endian signed int。
        """
        self._need(4)
        v = struct.unpack_from("<i", self.b, self.i)[0]
        self.i += 4
        return v

    def i64(self) -> int:
        """
        讀取 8 bytes little-endian signed int。
        """
        self._need(8)
        v = struct.unpack_from("<q", self.b, self.i)[0]
        self.i += 8
        return v

    def f32(self) -> float:
        """
        讀取 4 bytes little-endian float。
        """
        self._need(4)
        v = struct.unpack_from("<f", self.b, self.i)[0]
        self.i += 4
        return v

    def f64(self) -> float:
        """
        讀取 8 bytes little-endian double。
        PhotonVision 大多核心欄位(yaw / pitch / area / skew / ambiguity)
        都是用 f64。
        """
        self._need(8)
        v = struct.unpack_from("<d", self.b, self.i)[0]
        self.i += 8
        return v


def _skip_transform3d(buf: Buf):
    """
    跳過一個 Transform3d。
    Transform3d = Translation3d(3 doubles) + Rotation3d(4 doubles)
                = 7 doubles
                = 7 * 8 bytes
                = 56 bytes
    無使用 bestCameraToTarget / altCameraToTarget ，
    **但仍然必須跳過，避免後面的資料整包錯位。
    """
    buf._need(56)
    buf.i += 56


def _read_corner(buf: Buf) -> Corner:
    """
    讀取 TargetCorner。

    TargetCorner 格式：
    - float64 x
    - float64 y

    """
    return (buf.f64(), buf.f64())


def _read_corner_list(buf: Buf) -> List[Corner]:
    """
    讀取 corners。

    PhotonVision 列表格式：
    - 先讀 1 byte 長度 n
    - 再讀 n 個 corner

    因此角點數量不是硬寫死的 4,而是以封包裡的 n 為準。
    """
    n = buf.u8()
    return [_read_corner(buf) for _ in range(n)]


def _read_metadata(buf: Buf) -> Dict[str, int]:
    """
    讀取 PhotonPipelineMetadata。

    metadata 有 4 個 int64：
    - sequenceID
    - captureTimestampMicros
    - publishTimestampMicros
    - timeSinceLastPong

    這些資訊通用於：
    - 判斷新舊 frame(sequenceID)
    - 估計時間延遲
    - 檢查通訊狀態
    """
    return {
        "sequenceID": buf.i64(),
        "captureTimestampMicros": buf.i64(),
        "publishTimestampMicros": buf.i64(),
        "timeSinceLastPong": buf.i64(),
    }


def _read_target(buf: Buf) -> Dict[str, Any]:
    """
    讀取 PhotonTrackedTarget。

    你目前主要會用到的欄位有：
    - yaw
    - pitch
    - area
    - skew
    - fiducialId
    - objDetectId
    - objDetectConf
    - poseAmbiguity
    - minAreaRectCorners
    - detectedCorners

    """

    # -----------------------------
    # 基本 target 資料
    # -----------------------------
    yaw = buf.f64()
    pitch = buf.f64()
    area = buf.f64()
    skew = buf.f64()

    fid = buf.i32()

    # objDetectId 物件辨識類別 ID
    oid = buf.i32()

    # 物件辨識信心值
    conf = buf.f32()

    # -----------------------------
    # 兩個 Transform3d
    # -----------------------------
    # 必須正確跳過，讓後面的讀取位置維持正確。
    _skip_transform3d(buf)  # bestCameraToTarget
    _skip_transform3d(buf)  # altCameraToTarget

    # -----------------------------
    # pose ambiguity
    # -----------------------------
    amb = buf.f64()

    # -----------------------------
    # 角點資料
    # -----------------------------
    # minAreaRectCorners:
    #   最小外接旋轉矩形的角點列表
    #
    # detectedCorners:
    #   實際偵測到的目標角點列表
    min_rect_corners = _read_corner_list(buf)
    detected_corners = _read_corner_list(buf)

    # -----------------------------
    # 回傳整理好的 target dict
    # -----------------------------
    return {
        "yaw": yaw,
        "pitch": pitch,
        "area": area,
        "skew": skew,
        "fiducialId": fid,
        "objDetectId": oid,
        "objDetectConf": conf,
        "poseAmbiguity": amb,
        "minAreaRectCorners": min_rect_corners,
        "detectedCorners": detected_corners,
    }


def decode_pipeline_result(
    raw: bytes,
) -> Tuple[Dict[str, int], List[Dict[str, Any]], int, int]:

    buf = Buf(raw)

    # -----------------------------
    # 1. 先讀 metadata
    # -----------------------------
    md = _read_metadata(buf)

    # -----------------------------
    # 2. 讀取 target 數量
    # -----------------------------
    n_targets = buf.u8()

    # -----------------------------
    # 3. 逐一讀取所有 targets
    # -----------------------------
    targets = [_read_target(buf) for _ in range(n_targets)]

    # -----------------------------
    # 4. 嘗試讀 multitag flag
    # -----------------------------
    multitag_present = 0
    if buf.remaining() > 0:
        multitag_present = buf.u8()
        # multitag payload 目前不解析，先保留剩餘 bytes

    # -----------------------------
    # 5. 回傳結果
    # -----------------------------
    return md, targets, multitag_present, buf.remaining()