import cv2
import numpy as np
from collections import deque
from typing import List, Optional
from ..math_engine.types import Keypoint2d


class BarTracker:
    """
    Detecta la posicion de la barra del barbell dentro de los frames del video.

    Busca la barra (estructura metalica alargada y delgada) mediante el
    seguimiento de bordes con Canny y la deteccion de lineas con Hough.
    Devuelve la coordenada (x, y) del centroide de la barra detectada.

    Si no se logra detectar la barra en un frame, retorna `None` para que el
    llamador decida el comportamiento (por ejemplo, usar un proxy alternativo).

    Ademas detecta los discos (placas) del barbell en el frame.

    La posicion reportada pasa por un filtro Kalman (estado: posicion + velocidad)
    para suavizar la deteccion y evitar saltos grandes entre frames consecutivos.
    """

    SMOOTHING_WINDOW = 5

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self._history = deque(maxlen=self.SMOOTHING_WINDOW)
        self._kalman_state = np.zeros(4)  # [x, y, vx, vy]
        self._kalman_P = np.eye(4) * 10.0
        self._kalman_Q = np.eye(4) * 0.01
        self._kalman_R = np.eye(2) * 25.0
        self._dt = 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

    def _kalman(self, meas: np.ndarray) -> np.ndarray:
        x_pred = np.array([
            self._kalman_state[0] + self._kalman_state[2] * self._dt,
            self._kalman_state[1] + self._kalman_state[3] * self._dt,
            self._kalman_state[2],
            self._kalman_state[3],
        ])
        P_pred = self._kalman_P + self._kalman_Q
        H = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        y = meas - H @ x_pred
        S = H @ P_pred @ H.T + self._kalman_R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred
        self._kalman_state = x
        self._kalman_P = P
        return x

    def detect_barbell(self, frame: np.ndarray) -> Optional[Keypoint2d]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(
            edges,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=2,
        )

        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=30,
            minLineLength=25,
            maxLineGap=5,
        )

        if lines is None:
            return None

        best_line = None
        best_length = 0.0

        for line in lines:
            for x1, y1, x2, y2 in line.reshape(-1, 4).astype(float):
                length = float(np.hypot(x2 - x1, y2 - y1))
                if length > best_length:
                    best_length = length
                    best_line = (float(x1), float(y1), float(x2), float(y2))

        #El threshold de longitud evita aceptar lineas cortas/artefactos
        if best_line is None or best_length < 30.0:
            return None

        x1, y1, x2, y2 = best_line
        meas = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
        state = self._kalman(meas)
        barbell = Keypoint2d(x=float(state[0]), y=float(state[1]))

        self._history.append(barbell)
        return barbell

    def detect_plates(self, frame: np.ndarray) -> Optional[List[Keypoint2d]]:
        """
        Detecta los discos (placas) del barbell en el frame.
        Retorna una lista de puntos (x, y) con el centroide de cada disco.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.dilate(
            thresh,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=2,
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plates: List[Keypoint2d] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 30.0:
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            circularity = area / (perimeter * perimeter)
            if circularity > 0.6:
                x, y, w, h = cv2.boundingRect(contour)
                plates.append(Keypoint2d(x=float(x + w / 2.0), y=float(y + h / 2.0)))

        return plates if plates else None

    def procces_video(self, video_path: str) -> List[Keypoint2d]:
        """
        Lee el video frame a frame y retorna la lista de posiciones
        de la barra del barbell (x, y en pixeles) por frame.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30.0

        barbell_positions: List[Keypoint2d] = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            barbell = self.detect_barbell(frame)
            if barbell is not None:
                barbell_positions.append(barbell)

            frame_idx += 1

        cap.release()
        return barbell_positions
