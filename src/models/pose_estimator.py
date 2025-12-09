"""YOLO-11 pose estimation module."""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional

from ..utils.logger import logger
from ..utils.config import Config


class PoseEstimator:
    """YOLO-11 pose estimation for person detection."""

    # YOLO pose keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4

    def __init__(self):
        """Initialize YOLO-11 pose model."""
        logger.info(f"Loading YOLO-11 pose model on {Config.DEVICE}")
        self.model = YOLO('yolo11n-pose.pt')  # Using nano model for speed
        self.model.to(Config.DEVICE)

    def detect_poses(self, frame: np.ndarray) -> List[Dict]:
        """Detect poses in a frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detected persons with pose information
        """
        results = self.model(frame, verbose=False, device=Config.DEVICE)

        persons = []
        for result in results:
            if result.keypoints is None:
                continue

            boxes = result.boxes
            keypoints = result.keypoints

            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])

                # Filter by threshold
                if confidence < Config.POSE_THRESHOLD:
                    continue

                # Get bounding box
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, box)

                # Get keypoints
                kpts = keypoints.xy[i].cpu().numpy()  # (num_keypoints, 2)
                kpts_conf = keypoints.conf[i].cpu().numpy()  # (num_keypoints,)

                # Extract key facial keypoints
                nose = self._get_keypoint(kpts, kpts_conf, self.NOSE)
                left_eye = self._get_keypoint(kpts, kpts_conf, self.LEFT_EYE)
                right_eye = self._get_keypoint(kpts, kpts_conf, self.RIGHT_EYE)

                person = {
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2),
                    'keypoints': kpts,
                    'keypoints_conf': kpts_conf,
                    'nose': nose,
                    'left_eye': left_eye,
                    'right_eye': right_eye,
                }

                persons.append(person)

        return persons

    def _get_keypoint(self, kpts: np.ndarray, kpts_conf: np.ndarray,
                     idx: int) -> Optional[Tuple[int, int]]:
        """Get a specific keypoint if confidence is high enough.

        Args:
            kpts: Keypoints array
            kpts_conf: Keypoints confidence array
            idx: Keypoint index

        Returns:
            (x, y) tuple or None if confidence too low
        """
        if idx >= len(kpts) or kpts_conf[idx] < 0.5:
            return None

        x, y = kpts[idx]
        if x == 0 and y == 0:
            return None

        return (int(x), int(y))

    def get_eye_center(self, person: Dict) -> Optional[Tuple[int, int]]:
        """Calculate center point between eyes.

        Args:
            person: Person detection dict

        Returns:
            (x, y) center point or None
        """
        left_eye = person.get('left_eye')
        right_eye = person.get('right_eye')

        if left_eye is None or right_eye is None:
            return None

        center_x = (left_eye[0] + right_eye[0]) // 2
        center_y = (left_eye[1] + right_eye[1]) // 2

        return (center_x, center_y)

    def analyze_scene(self, video_path: str, start_frame: int = 0,
                     end_frame: int = -1) -> List[List[Dict]]:
        """Analyze all frames in a scene.

        Args:
            video_path: Path to video file
            start_frame: Starting frame number
            end_frame: Ending frame number (-1 for end of video)

        Returns:
            List of person detections for each frame
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame == -1:
            end_frame = total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        scene_detections = []

        for frame_num in range(start_frame, min(end_frame, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            persons = self.detect_poses(frame)
            scene_detections.append(persons)

        cap.release()

        return scene_detections
