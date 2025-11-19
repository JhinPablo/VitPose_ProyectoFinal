import platform
import time
from typing import Generator, List, Optional, Tuple, Union

import cv2


IS_WINDOWS = platform.system().lower().startswith("win")


class WebcamStream:
    """Gestiona la captura continua desde la webcam con OpenCV."""

    def __init__(
        self,
        index: int = 0,
        *,
        source: Union[int, str, None] = None,
        resolution: Optional[Tuple[int, int]] = None,
        fps: int = 30,
        backend: Optional[int] = None,
    ) -> None:
        self.source: Union[int, str] = source if source is not None else index
        self.resolution = resolution
        self.fps = fps
        self.backend = backend
        self._cap: Optional[cv2.VideoCapture] = None

    @staticmethod
    def _open_capture(source: Union[int, str], api: Optional[int]) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(source) if api is None else cv2.VideoCapture(source, api)
        if cap.isOpened():
            return cap
        cap.release()
        return None

    def _candidate_backends(self) -> List[Tuple[Union[int, str], Optional[int]]]:
        if isinstance(self.source, str):
            return [(self.source, None)]  # rutas de vídeo usan API por defecto

        ordered: List[Tuple[Union[int, str], Optional[int]]] = []
        if self.backend is not None:
            ordered.append((self.source, self.backend))
        else:
            if IS_WINDOWS:
                ordered.extend(
                    [
                        (self.source, cv2.CAP_DSHOW),
                        (self.source, cv2.CAP_MSMF),
                    ]
                )
            ordered.append((self.source, None))  # CAP_ANY
        return ordered

    def open(self) -> None:
        for src, api in self._candidate_backends():
            cap = self._open_capture(src, api)
            if cap is not None:
                self._cap = cap
                break

        if self._cap is None:
            raise RuntimeError(
                "No se pudo abrir la fuente de video. Si estás en Docker, expón la cámara con --device o usa un archivo de video."
            )

        if self.resolution:
            width, height = self.resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if self.fps:
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self):
        if self._cap is None:
            self.open()
        assert self._cap is not None
        success, frame = self._cap.read()
        if not success:
            raise RuntimeError("No se pudo capturar el frame")
        return frame

    def stream(self) -> Generator:
        frame_interval = 1.0 / self.fps if self.fps else 0
        while True:
            start = time.time()
            frame = self.read()
            yield frame
            elapsed = time.time() - start
            if frame_interval > 0 and elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)


def get_frame() -> "cv2.typing.MatLike":
    """Función utilitaria para capturar un solo frame."""
    stream = WebcamStream()
    try:
        stream.open()
        return stream.read()
    finally:
        stream.close()
