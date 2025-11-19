from typing import Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - requerido para proyección 3D

# Conexiones esqueléticas COCO para visualización rápida
COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (5, 6), (11, 12)
]


def plot_skeleton(joints, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(joints[:, 0], joints[:, 1], c='r')
    for i, j in COCO_SKELETON:
        if i < len(joints) and j < len(joints):
            ax.plot([joints[i, 0], joints[j, 0]], [joints[i, 1], joints[j, 1]], c='b')
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.invert_yaxis()
    ax.set_title('Modelo 3D (simulado en 2D)')
    return ax


def draw_skeleton_on_frame(frame: np.ndarray, joints: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    if joints.size == 0:
        return overlay
    for x, y in joints[:, :2]:
        cv2.circle(overlay, (int(x), int(y)), 4, (0, 0, 255), -1)
    for i, j in COCO_SKELETON:
        if i < len(joints) and j < len(joints):
            pt1 = tuple(np.int32(joints[i, :2]))
            pt2 = tuple(np.int32(joints[j, :2]))
            cv2.line(overlay, pt1, pt2, (255, 0, 0), 2)
    return overlay


def plot_skeleton_3d(joints_3d: np.ndarray, ax: Optional[Axes3D] = None) -> plt.Figure:
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    ax.clear()
    if joints_3d.size == 0:
        ax.set_title("Esqueleto 3D (sin datos)")
        return fig
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c="r")
    for i, j in COCO_SKELETON:
        if i < len(joints_3d) and j < len(joints_3d):
            ax.plot(
                [joints_3d[i, 0], joints_3d[j, 0]],
                [joints_3d[i, 1], joints_3d[j, 1]],
                [joints_3d[i, 2], joints_3d[j, 2]],
                c="b",
            )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Esqueleto 3D (aproximado)")
    ax.set_box_aspect([1, 1, 0.6])
    return fig
