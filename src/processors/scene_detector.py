"""Scene detection module to split video into scenes."""

import cv2
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from typing import List, Tuple
import os

from ..utils.logger import logger
from ..utils.config import Config


class SceneDetector:
    """Detect and split video into scenes."""

    def __init__(self, video_path: str):
        """Initialize scene detector.

        Args:
            video_path: Path to input video file
        """
        self.video_path = video_path
        self.video_manager = VideoManager([video_path])
        self.scene_manager = SceneManager()
        self.scene_manager.add_detector(
            ContentDetector(threshold=Config.SCENE_THRESHOLD,
                          min_scene_len=Config.MIN_SCENE_LENGTH)
        )

    def detect_scenes(self) -> List[Tuple[int, int]]:
        """Detect scenes in the video.

        Returns:
            List of tuples (start_frame, end_frame) for each scene
        """
        logger.info(f"Detecting scenes in {self.video_path}")

        self.video_manager.set_downscale_factor()
        self.video_manager.start()

        # Detect scenes
        self.scene_manager.detect_scenes(frame_source=self.video_manager)
        scene_list = self.scene_manager.get_scene_list()

        self.video_manager.release()

        # Convert to frame numbers
        scenes = []
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            scenes.append((start_frame, end_frame))
            logger.info(f"Scene {i+1}: frames {start_frame} to {end_frame}")

        if not scenes:
            # If no scenes detected, treat entire video as one scene
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            scenes = [(0, total_frames)]
            logger.info(f"No scene changes detected. Processing entire video as one scene (0-{total_frames})")

        return scenes

    def split_video_into_scenes(self, output_dir: str) -> List[str]:
        """Split video into separate scene files.

        Args:
            output_dir: Directory to save scene video files

        Returns:
            List of paths to scene video files
        """
        scenes = self.detect_scenes()
        scene_paths = []

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for i, (start_frame, end_frame) in enumerate(scenes):
            scene_path = os.path.join(output_dir, f"scene_{i:04d}.mp4")
            scene_paths.append(scene_path)

            logger.info(f"Extracting scene {i+1}/{len(scenes)}: {scene_path}")

            # Set position to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(scene_path, fourcc, fps, (width, height))

            # Write frames
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

            out.release()

        cap.release()
        logger.info(f"Extracted {len(scene_paths)} scenes")

        return scene_paths
