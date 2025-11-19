import json
import socket
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass
class PosePacket:
    keypoints: Iterable[Iterable[float]]
    scores: Iterable[float]
    joints_3d: Optional[Iterable[Iterable[float]]] = None
    timestamp: float = field(default_factory=time.time)
    meta: Optional[Dict[str, Any]] = None

    def to_json(self) -> str:
        payload = asdict(self)
        payload["keypoints"] = np.asarray(self.keypoints, dtype=float).tolist()
        payload["scores"] = np.asarray(self.scores, dtype=float).tolist()
        if self.joints_3d is not None:
            payload["joints_3d"] = np.asarray(self.joints_3d, dtype=float).tolist()
        if payload.get("meta") is not None:
            payload["meta"] = _serialize_meta(payload["meta"])
        return json.dumps(payload)


def _serialize_meta(meta: Any):
    if isinstance(meta, dict):
        return {k: _serialize_meta(v) for k, v in meta.items()}
    if isinstance(meta, (list, tuple)):
        return [_serialize_meta(v) for v in meta]
    if isinstance(meta, np.ndarray):
        return meta.tolist()
    return meta


def send_packet_udp(packet: PosePacket, host: str = "127.0.0.1", port: int = 5005) -> None:
    data = packet.to_json().encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, (host, port))


def send_keypoints_udp(keypoints: np.ndarray, host: str = "127.0.0.1", port: int = 5005) -> None:
    packet = PosePacket(keypoints=keypoints, scores=np.ones(len(keypoints)))
    send_packet_udp(packet, host, port)
