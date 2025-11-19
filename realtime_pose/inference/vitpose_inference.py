from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from mmpose.apis import inference_top_down_pose_model, init_pose_model
from mmpose.datasets import DatasetInfo


@dataclass
class PoseResult:
    keypoints: np.ndarray  # (K, 2)
    scores: np.ndarray  # (K,)
    meta: dict


class ViTPoseEstimator:
    """Wrapper liviano para ejecutar inferencia con ViTPose."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0") -> None:
        self.model = init_pose_model(config_path, checkpoint_path, device=device)
        self.num_keypoints = int(self.model.cfg.model.keypoint_head.out_channels)
        dataset = self.model.cfg.data.get("test", {})
        dataset_info_cfg = dataset.get("dataset_info")
        if isinstance(dataset_info_cfg, dict) and dataset_info_cfg:
            self.dataset_info = DatasetInfo(dataset_info_cfg)
            self.dataset_name = self.dataset_info.dataset_name
        else:
            self.dataset_info = None
            self.dataset_name = dataset.get("type", "TopDownCocoDataset")

    @staticmethod
    def _default_person_result(frame: np.ndarray) -> List[dict]:
        height, width = frame.shape[:2]
        return [dict(bbox=np.array([0, 0, width - 1, height - 1], dtype=np.float32))]

    def infer(self, frame: np.ndarray, person_results: Optional[List[dict]] = None) -> PoseResult:
        if person_results is None:
            person_results = self._default_person_result(frame)

        pose_results = inference_top_down_pose_model(
            self.model,
            frame,
            person_results,
            bbox_thr=0.3,
            format="xyxy",
            dataset=self.dataset_name,
            dataset_info=self.dataset_info,
            return_heatmap=False,
        )

        if not pose_results:
            height, width = frame.shape[:2]
            return PoseResult(
                keypoints=np.zeros((self.num_keypoints, 2), dtype=np.float32),
                scores=np.zeros(self.num_keypoints, dtype=np.float32),
                meta={"frame_size": (width, height)},
            )

        best = max(pose_results, key=lambda item: float(item.get("score", 0.0)))
        keypoints = best["keypoints"][:, :2]
        scores = best["keypoints"][:, 2]
        return PoseResult(keypoints=keypoints, scores=scores, meta=best)


def infer_pose(frame: np.ndarray, estimator: ViTPoseEstimator) -> PoseResult:
    return estimator.infer(frame)
