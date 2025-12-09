"""Main pipeline for converting landscape videos to portrait."""

import os
import cv2
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .models.pose_estimator import PoseEstimator
from .models.face_detector import FaceDetector
from .processors.scene_detector import SceneDetector
from .processors.crop_calculator import CropCalculator
from .processors.video_stitcher import VideoStitcher
from .utils.visualizer import Visualizer
from .utils.logger import logger
from .utils.config import Config


class PortraitConverter:
    """Main pipeline for converting landscape podcast videos to portrait."""

    def __init__(self, debug_mode: bool = False):
        """Initialize the converter.

        Args:
            debug_mode: Enable debug visualizations
        """
        self.debug_mode = debug_mode
        Config.set_debug(debug_mode)

        logger.info(f"Initializing PortraitConverter (GPU: {Config.DEVICE})")

        self.pose_estimator = PoseEstimator()
        self.face_detector = FaceDetector()
        self.crop_calculator = CropCalculator()

    def process_scene(self, scene_path: str, scene_idx: int, output_dir: str,
                     debug_dir: Optional[str] = None) -> Optional[str]:
        """Process a single scene.

        Args:
            scene_path: Path to scene video
            scene_idx: Scene index
            output_dir: Output directory for portrait video
            debug_dir: Debug frames directory

        Returns:
            Path to portrait video or None
        """
        logger.info(f"Processing scene {scene_idx}: {scene_path}")

        # Get video info
        cap = cv2.VideoCapture(scene_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Step 1: Detect all persons in all frames using YOLO
        logger.info("Step 1: Running pose estimation...")
        persons_per_frame = []

        cap = cv2.VideoCapture(scene_path)
        for frame_num in tqdm(range(total_frames), desc="Pose estimation"):
            ret, frame = cap.read()
            if not ret:
                break

            persons = self.pose_estimator.detect_poses(frame)
            persons_per_frame.append(persons)

        cap.release()

        # Analyze detection results
        num_persons = self._count_unique_persons(persons_per_frame)
        logger.info(f"Detected {num_persons} unique person(s) in scene")

        # Determine processing strategy
        if num_persons == 0:
            logger.warning("No persons detected in scene. Using center crop fallback.")
            return self._process_no_person_scene(scene_path, scene_idx, output_dir, debug_dir)

        elif num_persons == 1:
            # Single person: focus on them
            return self._process_single_person_scene(
                scene_path, scene_idx, persons_per_frame, output_dir, debug_dir
            )

        else:
            # Multiple persons: detect faces and analyze talking
            return self._process_multi_person_scene(
                scene_path, scene_idx, persons_per_frame, output_dir, debug_dir
            )

    def _count_unique_persons(self, persons_per_frame: List[List[Dict]]) -> int:
        """Estimate number of unique persons across frames.

        Args:
            persons_per_frame: List of person detections per frame

        Returns:
            Estimated number of unique persons
        """
        max_persons = 0
        for persons in persons_per_frame:
            if len(persons) > max_persons:
                max_persons = len(persons)

        return max_persons

    def _process_no_person_scene(self, scene_path: str, scene_idx: int,
                                 output_dir: str,
                                 debug_dir: Optional[str] = None) -> Optional[str]:
        """Process scene with no person detection using center crop.

        Args:
            scene_path: Scene video path
            scene_idx: Scene index
            output_dir: Output directory
            debug_dir: Debug directory

        Returns:
            Portrait video path
        """
        logger.info("No persons detected - using center crop")

        # Calculate center crop
        crop_region = self.crop_calculator.calculate_center_crop(scene_path)

        if crop_region is None:
            logger.error("Failed to calculate center crop region")
            return None

        # Apply crop
        output_path = os.path.join(output_dir, f"portrait_scene_{scene_idx:04d}.mp4")
        success = self.crop_calculator.apply_crop(scene_path, output_path, crop_region)

        if not success:
            return None

        # Save debug frames if enabled
        if self.debug_mode and debug_dir:
            # No detections to visualize, just save crop region
            from .utils.visualizer import Visualizer
            cap = cv2.VideoCapture(scene_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample a few frames to show center crop
            for frame_num in range(0, total_frames, Config.DEBUG_FRAME_INTERVAL):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                if not ret:
                    break

                # Draw crop region
                vis_frame = Visualizer.draw_crop_region(frame.copy(), crop_region)
                debug_path = os.path.join(debug_dir, f"scene_{scene_idx:04d}_frame_{frame_num:06d}_center_crop.jpg")
                cv2.imwrite(debug_path, vis_frame)

            cap.release()

        return output_path

    def _process_single_person_scene(self, scene_path: str, scene_idx: int,
                                    persons_per_frame: List[List[Dict]],
                                    output_dir: str,
                                    debug_dir: Optional[str] = None) -> Optional[str]:
        """Process scene with single person.

        Args:
            scene_path: Scene video path
            scene_idx: Scene index
            persons_per_frame: Person detections
            output_dir: Output directory
            debug_dir: Debug directory

        Returns:
            Portrait video path
        """
        logger.info("Single person detected - cropping to keep eyes/nose in frame")

        # Extract target person from each frame
        target_detections = []
        for persons in persons_per_frame:
            if persons:
                # Take first (and only) person
                target_detections.append(persons[0])
            else:
                target_detections.append(None)

        # Calculate static crop
        crop_region = self.crop_calculator.calculate_static_crop(
            scene_path, target_detections
        )

        if crop_region is None:
            logger.error("Failed to calculate crop region")
            return None

        # Apply crop
        output_path = os.path.join(output_dir, f"portrait_scene_{scene_idx:04d}.mp4")
        success = self.crop_calculator.apply_crop(scene_path, output_path, crop_region)

        if not success:
            return None

        # Save debug frames
        if self.debug_mode and debug_dir:
            faces_per_frame = [[] for _ in persons_per_frame]  # No faces detected
            Visualizer.save_debug_frames_for_scene(
                scene_path, debug_dir, scene_idx,
                persons_per_frame, faces_per_frame,
                crop_region, "YOLO_single",
                interval=Config.DEBUG_FRAME_INTERVAL
            )

        return output_path

    def _process_multi_person_scene(self, scene_path: str, scene_idx: int,
                                   persons_per_frame: List[List[Dict]],
                                   output_dir: str,
                                   debug_dir: Optional[str] = None) -> Optional[str]:
        """Process scene with multiple persons.

        Args:
            scene_path: Scene video path
            scene_idx: Scene index
            persons_per_frame: Person detections
            output_dir: Output directory
            debug_dir: Debug directory

        Returns:
            Portrait video path
        """
        logger.info("Multiple persons detected - analyzing faces and lip movement")

        # Step 2: Detect faces using RetinaFace
        logger.info("Step 2: Running face detection...")
        faces_per_frame = []

        cap = cv2.VideoCapture(scene_path)
        total_frames = len(persons_per_frame)

        for frame_num in tqdm(range(total_frames), desc="Face detection"):
            ret, frame = cap.read()
            if not ret:
                break

            faces = self.face_detector.detect_faces(frame)
            faces_per_frame.append(faces)

        cap.release()

        # Step 3: Match persons with faces
        logger.info("Step 3: Matching persons with faces...")
        matched_persons_faces = self._match_persons_with_faces(
            persons_per_frame, faces_per_frame
        )

        if not matched_persons_faces:
            logger.warning("No confirmed faces found. Using highest confidence person.")
            return self._fallback_to_best_person(
                scene_path, scene_idx, persons_per_frame, output_dir, debug_dir
            )

        # Step 4: Analyze lip movement to find talking person
        logger.info("Step 4: Analyzing lip movement...")
        talking_idx = self.face_detector.find_talking_person(
            scene_path, matched_persons_faces
        )

        if talking_idx is None:
            logger.warning("No talking person detected. Using highest confidence person.")
            talking_idx = 0  # Default to first matched person

        # Get target person
        target_person, target_face = matched_persons_faces[talking_idx]

        # Extract target person detections for all frames
        target_detections = self._extract_target_person_across_frames(
            persons_per_frame, target_person
        )

        # Calculate static crop
        crop_region = self.crop_calculator.calculate_static_crop(
            scene_path, target_detections
        )

        if crop_region is None:
            logger.error("Failed to calculate crop region")
            return None

        # Apply crop
        output_path = os.path.join(output_dir, f"portrait_scene_{scene_idx:04d}.mp4")
        success = self.crop_calculator.apply_crop(scene_path, output_path, crop_region)

        if not success:
            return None

        # Save debug frames
        if self.debug_mode and debug_dir:
            Visualizer.save_debug_frames_for_scene(
                scene_path, debug_dir, scene_idx,
                persons_per_frame, faces_per_frame,
                crop_region, "talking_detection",
                talking_idx=talking_idx,
                interval=Config.DEBUG_FRAME_INTERVAL
            )

        return output_path

    def _match_persons_with_faces(self, persons_per_frame: List[List[Dict]],
                                 faces_per_frame: List[List[Dict]]) -> List[Tuple[Dict, Dict]]:
        """Match detected persons with faces across frames.

        Args:
            persons_per_frame: Person detections per frame
            faces_per_frame: Face detections per frame

        Returns:
            List of (person, face) tuples that match
        """
        # Find a frame with both persons and faces
        for persons, faces in zip(persons_per_frame, faces_per_frame):
            if not persons or not faces:
                continue

            matched = []
            for person in persons:
                for face in faces:
                    if self.face_detector.is_pose_in_face(person, face):
                        matched.append((person, face))
                        break

            if matched:
                return matched

        return []

    def _extract_target_person_across_frames(self, persons_per_frame: List[List[Dict]],
                                            target_person: Dict) -> List[Dict]:
        """Extract target person across all frames using bounding box matching.

        Args:
            persons_per_frame: All person detections
            target_person: Reference person to track

        Returns:
            List of target person detections per frame
        """
        target_bbox = target_person['bbox']
        target_detections = []

        for persons in persons_per_frame:
            if not persons:
                target_detections.append(None)
                continue

            # Find person with most similar bounding box
            best_match = None
            best_iou = 0.0

            for person in persons:
                iou = self._calculate_iou(target_bbox, person['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match = person

            # Use match if IoU is reasonable
            if best_iou > 0.3:
                target_detections.append(best_match)
            else:
                target_detections.append(None)

        return target_detections

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union for two bounding boxes.

        Args:
            bbox1: (x1, y1, x2, y2)
            bbox2: (x1, y1, x2, y2)

        Returns:
            IoU score
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _fallback_to_best_person(self, scene_path: str, scene_idx: int,
                                persons_per_frame: List[List[Dict]],
                                output_dir: str,
                                debug_dir: Optional[str] = None) -> Optional[str]:
        """Fallback: use person with highest confidence score.

        Args:
            scene_path: Scene video path
            scene_idx: Scene index
            persons_per_frame: Person detections
            output_dir: Output directory
            debug_dir: Debug directory

        Returns:
            Portrait video path
        """
        logger.info("Using fallback: selecting person with highest confidence")

        # Find person with highest average confidence
        best_person = None
        best_conf = 0.0

        for persons in persons_per_frame:
            for person in persons:
                if person['confidence'] > best_conf:
                    best_conf = person['confidence']
                    best_person = person

        if best_person is None:
            logger.error("No valid person found")
            return None

        # Extract target person across frames
        target_detections = self._extract_target_person_across_frames(
            persons_per_frame, best_person
        )

        # Calculate static crop
        crop_region = self.crop_calculator.calculate_static_crop(
            scene_path, target_detections
        )

        if crop_region is None:
            logger.error("Failed to calculate crop region")
            return None

        # Apply crop
        output_path = os.path.join(output_dir, f"portrait_scene_{scene_idx:04d}.mp4")
        success = self.crop_calculator.apply_crop(scene_path, output_path, crop_region)

        if not success:
            return None

        # Save debug frames
        if self.debug_mode and debug_dir:
            faces_per_frame = [[] for _ in persons_per_frame]
            Visualizer.save_debug_frames_for_scene(
                scene_path, debug_dir, scene_idx,
                persons_per_frame, faces_per_frame,
                crop_region, "YOLO_fallback",
                interval=Config.DEBUG_FRAME_INTERVAL
            )

        return output_path

    def convert(self, input_video: str, output_video: str) -> bool:
        """Convert landscape video to portrait.

        Args:
            input_video: Input landscape video path
            output_video: Output portrait video path

        Returns:
            True if successful
        """
        logger.info(f"Converting {input_video} to portrait format")

        # Create working directories
        work_dir = "temp_work"
        scenes_dir = os.path.join(work_dir, "scenes")
        portraits_dir = os.path.join(work_dir, "portraits")
        debug_dir = "debug_frames" if self.debug_mode else None

        os.makedirs(scenes_dir, exist_ok=True)
        os.makedirs(portraits_dir, exist_ok=True)

        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)

        try:
            # Step 1: Detect and split scenes
            logger.info("=" * 60)
            logger.info("STEP 1: Scene Detection")
            logger.info("=" * 60)

            scene_detector = SceneDetector(input_video)
            scene_paths = scene_detector.split_video_into_scenes(scenes_dir)

            # Step 2: Process each scene
            logger.info("=" * 60)
            logger.info("STEP 2: Processing Scenes")
            logger.info("=" * 60)

            portrait_paths = []
            for i, scene_path in enumerate(scene_paths):
                portrait_path = self.process_scene(
                    scene_path, i, portraits_dir, debug_dir
                )

                if portrait_path:
                    portrait_paths.append(portrait_path)
                else:
                    logger.warning(f"Failed to process scene {i}")

            if not portrait_paths:
                logger.error("No scenes were successfully processed")
                return False

            # Step 3: Stitch portrait videos
            logger.info("=" * 60)
            logger.info("STEP 3: Stitching Videos")
            logger.info("=" * 60)

            temp_output = os.path.join(work_dir, "stitched_no_audio.mp4")
            VideoStitcher.stitch_with_ffmpeg(portrait_paths, temp_output)

            # Step 4: Add audio from original video
            logger.info("=" * 60)
            logger.info("STEP 4: Adding Audio")
            logger.info("=" * 60)

            success = VideoStitcher.add_audio(temp_output, input_video, output_video)

            if success:
                logger.info("=" * 60)
                logger.info(f"SUCCESS! Portrait video saved to: {output_video}")
                logger.info("=" * 60)
            else:
                # If audio adding fails, at least save video without audio
                logger.warning("Audio addition failed. Saving video without audio.")
                import shutil
                shutil.copy(temp_output, output_video)

            return True

        except Exception as e:
            logger.error(f"Error during conversion: {e}", exc_info=True)
            return False
