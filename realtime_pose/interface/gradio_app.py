import os
import sys

import cv2
from realtime_pose.ik.ik_solver import solve_ik
    overlay = draw_skeleton_on_frame(frame, joints)
        btn.click(fn=process_frame, inputs=gr.Textbox(visible=False), outputs=output)
import os
import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from realtime_pose.capture.webcam_capture import get_frame
from realtime_pose.communication.keypoints_sender import PosePacket
from realtime_pose.inference.vitpose_inference import PoseResult, ViTPoseEstimator
from realtime_pose.ik.ik_solver import SimpleIKSolver
from realtime_pose.model3d.model3d import draw_skeleton_on_frame


def load_estimator(config_path: str, checkpoint_path: str, device: str) -> ViTPoseEstimator:
    return ViTPoseEstimator(config_path, checkpoint_path, device=device)


def process_frame(estimator: ViTPoseEstimator, solver: SimpleIKSolver) -> tuple[np.ndarray, str]:
    frame = get_frame()
    pose: PoseResult = estimator.infer(frame)
    ik_solution = solver.solve(pose.keypoints)
    if pose.scores.size:
        kp_overlay = np.column_stack([pose.keypoints, pose.scores])
    else:
        kp_overlay = pose.keypoints
    overlay = draw_skeleton_on_frame(frame, kp_overlay)
    packet = PosePacket(
        keypoints=pose.keypoints,
        scores=pose.scores,
        joints_3d=ik_solution.joints_3d,
        meta=pose.meta,
    )
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), packet.to_json()



    estimator = load_estimator(config_path, checkpoint_path, device)
    solver = SimpleIKSolver()

    with gr.Blocks(title="ViTPose Real-Time Demo") as demo:
        gr.Markdown("# Animación 3D con ViTPose (demo basada en Gradio)")
        with gr.Row():
            output = gr.Image(label="Frame con esqueleto")
            json_output = gr.JSON(label="Paquete JSON")
        demo.load(
            lambda: None,
            inputs=None,
            outputs=None,
        )
        capture_button = gr.Button("Capturar frame")
        capture_button.click(
            lambda: process_frame(estimator, solver),
            inputs=None,
            outputs=[output, json_output],
        )
    return demo



    config = os.environ.get(
        "VITPOSE_CONFIG",
        str(PROJECT_ROOT / "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_simple_coco_256x192.py"),
    )
    checkpoint = os.environ.get("VITPOSE_CHECKPOINT", str(PROJECT_ROOT / "checkpoints/vitpose-l.pth"))
    device = os.environ.get("VITPOSE_DEVICE", "cuda:0")
    demo = build_app(config, checkpoint, device)
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
