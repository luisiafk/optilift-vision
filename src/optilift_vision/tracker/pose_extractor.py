import cv2
import mediapipe as mp
from typing import List
from ..math_engine.types import Keypoint2d, FramePose

class LiftTracker:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    def procces_video(self) -> List[FramePose]:
        """
        Reads the video frame by frame, exctract clever atomic points
        and return a cronologic list of FRAMEPOSES
        """

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 30.0

        frame_poses = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.procces(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks
                h, w, _ = frame.shape

                #C
                hip = Keypoint2d(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h,
                )

                knee = Keypoint2d(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h,
                )

                ankle = Keypoint2d(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h,
                )

                shoulder = Keypoint2d(
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h,
                )

                #Use the middle point beetween the wrists to aproximate the barbell
                wrist_l = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                wrist_r = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                barbell = Keypoint2d(
                    (wrist_l.x + wrist_r.x) / 2 * w,
                    (wrist_l.y + wrist_r.y) / 2 * h,
                )

                frame_poses.append(
                    FramePose(
                        frame_idx=frame_idx,
                        timestamp=timestamp,
                        barbell=barbell,
                        hip=hip,
                        knee=knee,
                        ankle=ankle,
                        shoulder=shoulder
                    )
                )
            frame_idx += 1

        cap.release()
        return frame_poses
