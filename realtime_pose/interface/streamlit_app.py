import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = PROJECT_ROOT / "resources/videos/default.mp4"
if str(PROJECT_ROOT) not in sys.path:  # ensure project root takes precedence for package resolution
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from realtime_pose.capture.webcam_capture import WebcamStream
from realtime_pose.communication.keypoints_sender import PosePacket, send_packet_udp
from realtime_pose.inference.vitpose_inference import PoseResult, ViTPoseEstimator
from realtime_pose.ik.ik_solver import SimpleIKSolver
from realtime_pose.model3d.model3d import draw_skeleton_on_frame, plot_skeleton_3d
from realtime_pose.utils.helpers import validate_path


st.set_page_config(page_title="ViTPose RTV", layout="wide")
st.title("Animación 3D en tiempo real con ViTPose")


def init_session_state():
    defaults = {
        "estimator": None,
        "webcam": None,
        "ik_solver": SimpleIKSolver(depth_scale=1.0),
        "run_stream": False,
        "last_pose": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_session_state()


def load_estimator(config_path: str, checkpoint_path: str, device: str) -> ViTPoseEstimator:
    config = validate_path(config_path)
    checkpoint = validate_path(checkpoint_path)
    st.session_state.estimator = ViTPoseEstimator(str(config), str(checkpoint), device=device)
    st.sidebar.success("Modelo ViTPose cargado correctamente")


def start_stream(index: int, width: int, height: int, fps: int, video_path: str = ""):
    stop_stream()
    source = video_path.strip() or index
    webcam = WebcamStream(index=index, source=source, resolution=(width, height), fps=fps)
    webcam.open()
    st.session_state.webcam = webcam
    st.session_state.run_stream = True


def stop_stream():
    stream: WebcamStream = st.session_state.get("webcam")
    if stream:
        stream.close()
    st.session_state.run_stream = False
    st.session_state.webcam = None


with st.sidebar:
    st.header("Configuración")
    config_path = st.text_input(
        "Config ViTPose",
        value=str(PROJECT_ROOT / "configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_simple_coco_256x192.py")
    )
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    available_checkpoints = sorted(checkpoint_dir.glob("*.pth")) if checkpoint_dir.exists() else []
    default_checkpoint = str(available_checkpoints[0]) if available_checkpoints else ""
    checkpoint_path = st.text_input("Checkpoint (.pth)", value=default_checkpoint, placeholder=str(checkpoint_dir / "vitpose-*.pth"))
    if not available_checkpoints:
        st.caption("Coloca tus pesos en `checkpoints/` o proporciona una ruta absoluta al archivo .pth")
    device = st.selectbox("Dispositivo", options=["cuda:0", "cpu"])
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Cargar modelo"):
            try:
                load_estimator(config_path, checkpoint_path, device)
            except Exception as exc:  # pragma: no cover - feedback runtime
                st.error(f"No se pudo cargar el modelo: {exc}")
    with col_b:
        if st.button("Liberar modelo"):
            st.session_state.estimator = None
            st.success("Modelo liberado")

    st.divider()

    cam_index = st.number_input("ID de webcam", min_value=0, value=0, step=1)
    resolution = st.selectbox("Resolución", options=["640x480", "1280x720"], index=0)
    width, height = map(int, resolution.split("x"))
    fps = st.slider("FPS objetivo", min_value=5, max_value=30, value=15)
    default_video_value = str(DEFAULT_VIDEO) if DEFAULT_VIDEO.exists() else ""
    video_path = st.text_input("Ruta de video (opcional)", value=default_video_value)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Iniciar stream"):
            try:
                start_stream(cam_index, width, height, fps, video_path)
            except Exception as exc:
                st.error(f"No se pudo iniciar la webcam: {exc}")
    with col2:
        if st.button("Detener stream"):
            stop_stream()

    st.divider()
    send_udp = st.checkbox("Enviar por UDP", value=False)
    udp_host = st.text_input("Host UDP", value="127.0.0.1")
    udp_port = st.number_input("Puerto UDP", min_value=1024, max_value=65535, value=5005)


placeholder_info = st.empty()
col_video, col_pose = st.columns(2)
video_placeholder = col_video.empty()
heatmap_placeholder = col_pose.empty()
info_placeholder = col_pose.empty()


def process_frame(frame: np.ndarray) -> PoseResult:
    estimator: ViTPoseEstimator = st.session_state.estimator
    if estimator is None:
        raise RuntimeError("Cargue el modelo ViTPose antes de iniciar el stream")
    return estimator.infer(frame)


def push_packets(pose: PoseResult, joints_3d: np.ndarray):
    if not st.session_state.get("send_udp"):
        return
    packet = PosePacket(
        keypoints=pose.keypoints,
        scores=pose.scores,
        joints_3d=joints_3d,
        timestamp=time.time(),
        meta=pose.meta,
    )
    send_packet_udp(packet, host=st.session_state["udp_host"], port=st.session_state["udp_port"])


st.session_state["send_udp"] = send_udp
st.session_state["udp_host"] = udp_host
st.session_state["udp_port"] = udp_port


def update_visualizations(frame: np.ndarray, pose: PoseResult, joints_3d: np.ndarray):
    if pose.scores.size:
        kp_overlay = np.hstack([pose.keypoints, pose.scores[:, None]])
    else:
        kp_overlay = np.hstack([pose.keypoints, np.ones((len(pose.keypoints), 1))]) if len(pose.keypoints) else pose.keypoints
    overlay = draw_skeleton_on_frame(frame, kp_overlay)
    video_placeholder.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Stream con esqueleto", use_column_width=True)

    fig3d = plot_skeleton_3d(joints_3d)
    heatmap_placeholder.pyplot(fig3d)
    plt.close(fig3d)

    info_placeholder.write({
        "puntos": int(len(pose.keypoints)),
        "confianza_media": float(np.mean(pose.scores)) if pose.scores.size else 0.0,
    })


def process_stream_frame():
    webcam: WebcamStream = st.session_state.get("webcam")
    if webcam is None:
        placeholder_info.warning("Inicie la webcam desde la barra lateral")
        stop_stream()
        return

    if st.session_state.estimator is None:
        placeholder_info.warning("Cargue el modelo ViTPose para comenzar")
        time.sleep(0.5)
        st.experimental_rerun()
        return

    frame = webcam.read()
    pose = process_frame(frame)
    solver: SimpleIKSolver = st.session_state.ik_solver
    ik_solution = solver.solve(pose.keypoints)
    st.session_state.last_pose = (pose, ik_solution)
    update_visualizations(frame, pose, ik_solution.joints_3d)
    if send_udp:
        push_packets(pose, ik_solution.joints_3d)
    if st.session_state.run_stream:
        time.sleep(1.0 / st.session_state.webcam.fps)
        st.experimental_rerun()


if st.session_state.run_stream:
    process_stream_frame()
else:
    placeholder_info.info("Configure el modelo y presione 'Iniciar stream' para comenzar")


if st.session_state.last_pose:
    pose, ik_solution = st.session_state.last_pose
    st.subheader("Último paquete JSON")
    packet = PosePacket(
        keypoints=pose.keypoints,
        scores=pose.scores,
        joints_3d=ik_solution.joints_3d,
        timestamp=time.time(),
        meta=pose.meta,
    )
    st.code(packet.to_json(), language="json")
