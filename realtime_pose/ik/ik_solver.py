from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from ikpy.chain import Chain
    from ikpy.link import OriginLink, URDFLink
except ImportError:  # pragma: no cover - IKPy opcional
    Chain = None
    URDFLink = None  # type: ignore[assignment]


@dataclass
class IKSolution:
    joints_3d: np.ndarray  # (K, 3)
    keypoints_2d: np.ndarray  # (K, 2)


class SimpleIKSolver:
    """Convierte keypoints 2D en un esqueleto 3D simple mediante normalización."""

    def __init__(self, depth_scale: float = 1.0) -> None:
        self.depth_scale = depth_scale
        self.chain: Optional[Chain] = self._build_chain() if Chain else None

    def _build_chain(self) -> Optional[Chain]:
        if Chain is None:
            return None
        links = [OriginLink()]

        kwargs_name = URDFLink.__init__.__code__.co_varnames  # type: ignore[attr-defined]
        uses_legacy_api = "translation_vector" in kwargs_name

        for idx in range(1, 6):
            if uses_legacy_api:
                link = URDFLink(  # type: ignore[call-arg]
                    name=f"joint_{idx}",
                    translation_vector=[0.0, 0.0, 0.02],
                    orientation=[0.0, 0.0, 0.0],
                    rotation=(0, 0, 1),
                )
            else:
                link = URDFLink(  # type: ignore[call-arg]
                    name=f"joint_{idx}",
                    origin_translation=np.array([0.0, 0.0, 0.02]),
                    origin_orientation=np.array([0.0, 0.0, 0.0]),
                    rotation=np.array([0.0, 0.0, 1.0]),
                )
            links.append(link)
        return Chain(name="vitpose_chain", links=links)

    def solve(self, keypoints: np.ndarray) -> IKSolution:
        if keypoints.size == 0:
            return IKSolution(joints_3d=np.zeros((0, 3)), keypoints_2d=keypoints)

        normalized = self._normalize_keypoints(keypoints)
        depth = self._estimate_depth(normalized)
        joints_3d = np.concatenate([normalized, depth[:, None]], axis=1) * self.depth_scale
        return IKSolution(joints_3d=joints_3d, keypoints_2d=keypoints)

    @staticmethod
    def _normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
        centroid = keypoints.mean(axis=0, keepdims=True)
        centered = keypoints - centroid
        scale = np.max(np.linalg.norm(centered, axis=1)) or 1.0
        return centered / scale

    @staticmethod
    def _estimate_depth(normalized_keypoints: np.ndarray) -> np.ndarray:
        y = normalized_keypoints[:, 1]
        depth = (y - y.min()) / (y.max() - y.min() + 1e-6)
        return depth * 0.5  # ajusta profundidad relativa


def solve_ik(keypoints: np.ndarray) -> IKSolution:
    solver = SimpleIKSolver(depth_scale=1.0)
    return solver.solve(keypoints)
