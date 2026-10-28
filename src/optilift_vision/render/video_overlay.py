import cv2
import numpy as np
from typing import List
from ..math_engine.types import FramePose, KinematicState

class VideoRenderer:
    def __init__(self, input_video_path: str, output_video_path: str):
        self.input_path = input_video_path
        self.output_path =output_video_path

    def render(
        self,
        poses: List[FramePose],
        states: List[KinematicState],
        sticking_point_frame: int = 0
    ):
        """
        Procces the video frame by frame:
        1. Historic trajectory of the barbell
        2. Telemetric in real time
        3. Visual alert of the sticky point if occurs on the frame
        """

        cap = cv2.VideoCapture(self.input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        #Config the video writer of the output (codec mp4v)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            #Trace the historic barbell trajectory until the actual frame
            if frame_idx < len(poses):
                for i in range(1, min(frame_idx + 1, len(poses))):
                    pt1 = (int(poses[i-1].barbell.x), int(poses[i-1].barbell.y))
                    pt2 = (int(poses[i].barbell.x), int(poses[i].barbell.y))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                    #Actual point of the barbell
                curr_bar = (int(poses[frame_idx].barbell.x), int(poses[frame_idx].barbell.y))
                cv2.circle(frame, curr_bar, 6, (0, 0, 255), -1)

                #Puntos de articulacion (articular points) detectadas en el frame actual
                for pt in (
                    poses[frame_idx].hip,
                    poses[frame_idx].knee,
                    poses[frame_idx].ankle,
                    poses[frame_idx].shoulder,
                ):
                    cv2.circle(frame, (int(pt.x), int(pt.y)), 5, (255, 255, 0), -1)

                #Discos (placas) detectados en el frame actual
                for plate in poses[frame_idx].plates or []:
                    cv2.circle(frame, (int(plate.x), int(plate.y)), 7, (0, 255, 255), -1)

            if frame_idx < len(states):
                st = states[frame_idx]
                text_vel = f"Vel Y: {st.vel_y:.2f} m/s"
                text_acc = f"Acc Y: {st.acc_y:.2f} m/s^2"

                cv2.putText(frame, text_vel, (40,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.putText(frame, text_acc, (40,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if sticking_point_frame != 0 and frame_idx == sticking_point_frame:
                if frame_idx < len(poses):
                    sp_pos = (int(poses[frame_idx].barbell.x), int(poses[frame_idx].barbell.y))
                    cv2.circle(frame, sp_pos, 20, (0, 0, 255), 4)
                    cv2.putText(frame, "STICKING POINT", (sp_pos[0] + 25, sp_pos[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            out.write(frame)
            frame_idx += 1
        cap.release()
        out.release()
