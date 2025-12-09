"""Visualization and debugging utilities."""

import cv2
import numpy as np
import os
from typing import List, Dict, Optional, Tuple

from .logger import logger
from .config import Config


class Visualizer:
    """Visualize detections for debugging."""

    @staticmethod
    def draw_pose(frame: np.ndarray, person: Dict, color=(0, 255, 0)) -> np.ndarray:
        """Draw pose keypoints on frame.

        Args:
            frame: Input frame
            person: Person detection dict
            color: BGR color tuple

        Returns:
            Frame with pose drawn
        """
        frame = frame.copy()

        # Draw bounding box
        bbox = person['bbox']
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw confidence
        conf_text = f"{person['confidence']:.2f}"
        cv2.putText(frame, conf_text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw keypoints
        keypoints = person['keypoints']
        keypoints_conf = person['keypoints_conf']

        for i, (kpt, conf) in enumerate(zip(keypoints, keypoints_conf)):
            if conf > 0.5:
                x, y = int(kpt[0]), int(kpt[1])
                if x > 0 and y > 0:
                    cv2.circle(frame, (x, y), 3, color, -1)

        # Highlight eyes and nose
        if person.get('nose'):
            cv2.circle(frame, person['nose'], 5, (0, 0, 255), -1)

        if person.get('left_eye'):
            cv2.circle(frame, person['left_eye'], 5, (255, 0, 0), -1)

        if person.get('right_eye'):
            cv2.circle(frame, person['right_eye'], 5, (255, 0, 0), -1)

        return frame

    @staticmethod
    def draw_face(frame: np.ndarray, face: Dict, color=(255, 0, 255)) -> np.ndarray:
        """Draw face detection on frame.

        Args:
            frame: Input frame
            face: Face detection dict
            color: BGR color tuple

        Returns:
            Frame with face drawn
        """
        frame = frame.copy()

        # Draw bounding box
        bbox = face['bbox']
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw confidence
        conf_text = f"{face['confidence']:.2f}"
        cv2.putText(frame, conf_text, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw landmarks
        if face.get('left_eye'):
            cv2.circle(frame, face['left_eye'], 3, (255, 255, 0), -1)

        if face.get('right_eye'):
            cv2.circle(frame, face['right_eye'], 3, (255, 255, 0), -1)

        if face.get('nose'):
            cv2.circle(frame, face['nose'], 3, (0, 255, 255), -1)

        if face.get('mouth_left'):
            cv2.circle(frame, face['mouth_left'], 3, (255, 0, 255), -1)

        if face.get('mouth_right'):
            cv2.circle(frame, face['mouth_right'], 3, (255, 0, 255), -1)

        return frame

    @staticmethod
    def draw_crop_region(frame: np.ndarray, crop_region: Tuple[int, int, int, int],
                        color=(0, 255, 255)) -> np.ndarray:
        """Draw crop region on frame.

        Args:
            frame: Input frame
            crop_region: (x, y, width, height)
            color: BGR color tuple

        Returns:
            Frame with crop region drawn
        """
        frame = frame.copy()

        x, y, w, h = crop_region
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        # Draw corner markers
        marker_size = 20
        # Top-left
        cv2.line(frame, (x, y), (x+marker_size, y), color, 5)
        cv2.line(frame, (x, y), (x, y+marker_size), color, 5)
        # Top-right
        cv2.line(frame, (x+w, y), (x+w-marker_size, y), color, 5)
        cv2.line(frame, (x+w, y), (x+w, y+marker_size), color, 5)
        # Bottom-left
        cv2.line(frame, (x, y+h), (x+marker_size, y+h), color, 5)
        cv2.line(frame, (x, y+h), (x, y+h-marker_size), color, 5)
        # Bottom-right
        cv2.line(frame, (x+w, y+h), (x+w-marker_size, y+h), color, 5)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-marker_size), color, 5)

        return frame

    @staticmethod
    def create_debug_frame(frame: np.ndarray, persons: List[Dict],
                          faces: List[Dict], crop_region: Optional[Tuple[int, int, int, int]],
                          detection_method: str, talking_idx: Optional[int] = None) -> np.ndarray:
        """Create comprehensive debug visualization.

        Args:
            frame: Input frame
            persons: List of person detections
            faces: List of face detections
            crop_region: Crop region tuple
            detection_method: Method used (YOLO, RetinaFace, or talking)
            talking_idx: Index of talking person

        Returns:
            Debug frame
        """
        debug_frame = frame.copy()

        # Draw all persons
        for i, person in enumerate(persons):
            color = (0, 255, 0) if talking_idx is None or i == talking_idx else (100, 100, 100)
            debug_frame = Visualizer.draw_pose(debug_frame, person, color)

        # Draw all faces
        for i, face in enumerate(faces):
            color = (255, 0, 255) if talking_idx is None or i == talking_idx else (150, 0, 150)
            debug_frame = Visualizer.draw_face(debug_frame, face, color)

        # Draw crop region
        if crop_region:
            debug_frame = Visualizer.draw_crop_region(debug_frame, crop_region)

        # Add text overlay
        y_offset = 30
        cv2.putText(debug_frame, f"Persons: {len(persons)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 40

        cv2.putText(debug_frame, f"Faces: {len(faces)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        y_offset += 40

        cv2.putText(debug_frame, f"Method: {detection_method}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        y_offset += 40

        if talking_idx is not None:
            cv2.putText(debug_frame, f"Talking: Person {talking_idx}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return debug_frame

    @staticmethod
    def save_debug_frame(frame: np.ndarray, output_dir: str, scene_idx: int,
                        frame_idx: int, persons: List[Dict], faces: List[Dict],
                        crop_region: Optional[Tuple[int, int, int, int]],
                        detection_method: str, talking_idx: Optional[int] = None):
        """Save debug frame to disk.

        Args:
            frame: Input frame
            output_dir: Output directory for debug frames
            scene_idx: Scene index
            frame_idx: Frame index within scene
            persons: Person detections
            faces: Face detections
            crop_region: Crop region
            detection_method: Detection method used
            talking_idx: Talking person index
        """
        os.makedirs(output_dir, exist_ok=True)

        debug_frame = Visualizer.create_debug_frame(
            frame, persons, faces, crop_region, detection_method, talking_idx
        )

        filename = f"scene_{scene_idx:04d}_frame_{frame_idx:06d}_{detection_method}.jpg"
        output_path = os.path.join(output_dir, filename)

        cv2.imwrite(output_path, debug_frame)

    @staticmethod
    def save_debug_frames_for_scene(video_path: str, output_dir: str, scene_idx: int,
                                   persons_per_frame: List[List[Dict]],
                                   faces_per_frame: List[List[Dict]],
                                   crop_region: Tuple[int, int, int, int],
                                   detection_method: str,
                                   talking_idx: Optional[int] = None,
                                   start_frame: int = 0,
                                   end_frame: int = -1,
                                   interval: int = 30):
        """Save debug frames for a scene at regular intervals.

        Args:
            video_path: Path to video
            output_dir: Output directory
            scene_idx: Scene index
            persons_per_frame: Person detections for each frame
            faces_per_frame: Face detections for each frame
            crop_region: Crop region
            detection_method: Detection method
            talking_idx: Talking person index
            start_frame: Start frame
            end_frame: End frame
            interval: Save every N frames
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame == -1:
            end_frame = total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for i, frame_num in enumerate(range(start_frame, min(end_frame, total_frames))):
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame at intervals
            if i % interval == 0:
                persons = persons_per_frame[i] if i < len(persons_per_frame) else []
                faces = faces_per_frame[i] if i < len(faces_per_frame) else []

                Visualizer.save_debug_frame(
                    frame, output_dir, scene_idx, frame_num,
                    persons, faces, crop_region, detection_method, talking_idx
                )

        cap.release()
        logger.info(f"Saved debug frames for scene {scene_idx} to {output_dir}")
