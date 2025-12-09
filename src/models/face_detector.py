"""RetinaFace face detection and lip movement analysis."""

import cv2
import numpy as np
from retinaface import RetinaFace
from typing import List, Dict, Tuple, Optional

from ..utils.logger import logger
from ..utils.config import Config


class FaceDetector:
    """RetinaFace detector with lip movement analysis."""

    def __init__(self):
        """Initialize RetinaFace detector."""
        logger.info("Initializing RetinaFace detector")

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces in a frame using RetinaFace.

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detected faces with landmarks
        """
        # RetinaFace expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        faces = RetinaFace.detect_faces(rgb_frame)

        if not isinstance(faces, dict):
            return []

        detected_faces = []
        for key, face_data in faces.items():
            confidence = face_data['score']

            # Filter by threshold
            if confidence < Config.FACE_THRESHOLD:
                continue

            # Get facial area (bounding box)
            facial_area = face_data['facial_area']
            x1, y1, x2, y2 = facial_area

            # Get landmarks
            landmarks = face_data['landmarks']
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            nose = landmarks['nose']
            mouth_left = landmarks['mouth_left']
            mouth_right = landmarks['mouth_right']

            face_dict = {
                'confidence': confidence,
                'bbox': (x1, y1, x2, y2),
                'left_eye': tuple(map(int, left_eye)),
                'right_eye': tuple(map(int, right_eye)),
                'nose': tuple(map(int, nose)),
                'mouth_left': tuple(map(int, mouth_left)),
                'mouth_right': tuple(map(int, mouth_right)),
                'landmarks': landmarks
            }

            detected_faces.append(face_dict)

        return detected_faces

    def is_pose_in_face(self, person: Dict, face: Dict) -> bool:
        """Check if pose keypoints (eyes, nose) fall within face bbox.

        Args:
            person: Person dict from pose estimation
            face: Face dict from face detection

        Returns:
            True if pose keypoints align with face
        """
        face_bbox = face['bbox']
        x1, y1, x2, y2 = face_bbox

        # Check nose
        nose = person.get('nose')
        if nose is not None:
            nx, ny = nose
            if not (x1 <= nx <= x2 and y1 <= ny <= y2):
                return False

        # Check eyes (at least one should be in bbox)
        left_eye = person.get('left_eye')
        right_eye = person.get('right_eye')

        eye_in_bbox = False
        if left_eye is not None:
            ex, ey = left_eye
            if x1 <= ex <= x2 and y1 <= ey <= y2:
                eye_in_bbox = True

        if right_eye is not None:
            ex, ey = right_eye
            if x1 <= ex <= x2 and y1 <= ey <= y2:
                eye_in_bbox = True

        return eye_in_bbox

    def get_mouth_region(self, frame: np.ndarray, face: Dict) -> Optional[np.ndarray]:
        """Extract mouth region from face.

        Args:
            frame: Input frame
            face: Face dict

        Returns:
            Mouth region image or None
        """
        mouth_left = face['mouth_left']
        mouth_right = face['mouth_right']
        nose = face['nose']

        # Calculate mouth region bbox
        mx1 = max(0, mouth_left[0] - 10)
        mx2 = min(frame.shape[1], mouth_right[0] + 10)
        my1 = max(0, nose[1])
        my2 = min(frame.shape[0], mouth_left[1] + 30)

        if mx2 <= mx1 or my2 <= my1:
            return None

        mouth_region = frame[my1:my2, mx1:mx2]
        return mouth_region

    def calculate_lip_movement(self, video_path: str, face_bbox: Tuple[int, int, int, int],
                               start_frame: int = 0, end_frame: int = -1,
                               sample_rate: int = 5) -> float:
        """Calculate lip movement variance for a face across frames.

        Args:
            video_path: Path to video file
            face_bbox: Face bounding box to track
            start_frame: Starting frame
            end_frame: Ending frame
            sample_rate: Sample every Nth frame

        Returns:
            Variance score (higher = more movement)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame == -1:
            end_frame = total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        mouth_regions = []
        x1, y1, x2, y2 = face_bbox

        # Expand bbox slightly for mouth region
        mouth_y1 = int(y1 + (y2 - y1) * 0.6)
        mouth_y2 = y2
        mouth_x1 = int(x1 + (x2 - x1) * 0.2)
        mouth_x2 = int(x2 - (x2 - x1) * 0.2)

        frame_count = 0
        for frame_num in range(start_frame, min(end_frame, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_rate == 0:
                # Extract mouth region
                mouth_region = frame[mouth_y1:mouth_y2, mouth_x1:mouth_x2]

                if mouth_region.size > 0:
                    # Convert to grayscale and resize for consistency
                    gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
                    resized_mouth = cv2.resize(gray_mouth, (50, 30))
                    mouth_regions.append(resized_mouth.flatten())

            frame_count += 1

        cap.release()

        if len(mouth_regions) < 2:
            return 0.0

        # Calculate variance across frames
        mouth_array = np.array(mouth_regions)
        variance = np.var(mouth_array, axis=0).mean()

        return float(variance)

    def extract_mouth_region_from_face(self, frame: np.ndarray, face: Dict) -> Optional[np.ndarray]:
        """Extract mouth region from the lower third of detected face.

        Args:
            frame: Input frame
            face: Face dict with bbox

        Returns:
            Grayscale mouth region or None
        """
        x1, y1, x2, y2 = face['bbox']

        # Extract lower third of face (66% down from top)
        # This region typically contains the mouth
        face_height = y2 - y1
        mouth_y1 = int(y1 + face_height * 0.66)
        mouth_y2 = y2

        # Ensure valid coordinates
        mouth_y1 = max(0, mouth_y1)
        mouth_y2 = min(frame.shape[0], mouth_y2)
        x1 = max(0, x1)
        x2 = min(frame.shape[1], x2)

        if mouth_y2 <= mouth_y1 or x2 <= x1:
            return None

        # Extract mouth region and convert to grayscale
        mouth_region = frame[mouth_y1:mouth_y2, x1:x2]
        gray_mouth = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)

        return gray_mouth

    def calculate_mouth_motion(self, prev_mouth: np.ndarray, curr_mouth: np.ndarray) -> float:
        """Calculate motion between two mouth regions using frame-to-frame difference.

        Args:
            prev_mouth: Previous frame's mouth region (grayscale)
            curr_mouth: Current frame's mouth region (grayscale)

        Returns:
            Motion score (0.0 to 1.0), higher = more movement
        """
        if prev_mouth is None or curr_mouth is None:
            return 0.0

        # Ensure both regions are same size
        if prev_mouth.shape != curr_mouth.shape:
            curr_mouth = cv2.resize(curr_mouth, (prev_mouth.shape[1], prev_mouth.shape[0]))

        # Calculate absolute difference between frames
        diff = cv2.absdiff(prev_mouth, curr_mouth)

        # Normalize to get motion score (0.0 to 1.0)
        total_pixels = diff.shape[0] * diff.shape[1] * 255
        motion = np.sum(diff) / total_pixels if total_pixels > 0 else 0.0

        return float(motion)

    def find_talking_person(self, video_path: str, persons_with_faces: List[Tuple[Dict, Dict]],
                           start_frame: int = 0, end_frame: int = -1) -> Optional[int]:
        """Identify which person is talking based on frame-to-frame mouth motion.

        Uses a voting system where each frame votes for the person with most mouth movement.
        The person with the most votes across the scene becomes the talking person.

        Args:
            video_path: Path to video file
            persons_with_faces: List of (person, face) tuples
            start_frame: Starting frame
            end_frame: Ending frame

        Returns:
            Index of talking person or None
        """
        if not persons_with_faces:
            return None

        logger.info(f"Analyzing mouth motion for {len(persons_with_faces)} person(s)")

        num_persons = len(persons_with_faces)

        # Voting system: track votes for each person
        motion_votes = [0] * num_persons
        total_frames_analyzed = 0

        # Motion threshold (2% pixel change)
        MOTION_THRESHOLD = 0.02

        # Open video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame == -1:
            end_frame = total_frames

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Store previous mouth regions for each person
        prev_mouth_regions = [None] * num_persons

        # Sample every 3rd frame for efficiency
        sample_rate = 3
        frame_count = 0

        for frame_num in range(start_frame, min(end_frame, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every Nth frame
            if frame_count % sample_rate != 0:
                frame_count += 1
                continue

            # Detect faces in current frame
            faces = self.detect_faces(frame)
            if not faces or len(faces) < num_persons:
                frame_count += 1
                continue

            # Calculate motion for each person
            motion_scores = []

            for idx, (person, reference_face) in enumerate(persons_with_faces):
                # Find matching face in current frame (using bbox overlap)
                matched_face = None
                best_overlap = 0.0

                for face in faces:
                    overlap = self._calculate_bbox_overlap(reference_face['bbox'], face['bbox'])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        matched_face = face

                if matched_face is None or best_overlap < 0.3:
                    motion_scores.append(0.0)
                    continue

                # Extract mouth region
                curr_mouth = self.extract_mouth_region_from_face(frame, matched_face)

                if curr_mouth is None:
                    motion_scores.append(0.0)
                    continue

                # Calculate motion if we have previous frame
                if prev_mouth_regions[idx] is not None:
                    motion = self.calculate_mouth_motion(prev_mouth_regions[idx], curr_mouth)
                    motion_scores.append(motion)
                else:
                    motion_scores.append(0.0)

                # Store for next iteration
                prev_mouth_regions[idx] = curr_mouth

            # Vote: person with highest motion in this frame gets a vote
            if motion_scores and max(motion_scores) > MOTION_THRESHOLD:
                talking_idx_this_frame = motion_scores.index(max(motion_scores))
                motion_votes[talking_idx_this_frame] += 1
                total_frames_analyzed += 1

            frame_count += 1

        cap.release()

        # Determine winner by votes
        if total_frames_analyzed > 0 and max(motion_votes) > 0:
            talking_idx = motion_votes.index(max(motion_votes))
            vote_percentage = (motion_votes[talking_idx] / total_frames_analyzed) * 100

            logger.info(f"Motion voting results:")
            for idx, votes in enumerate(motion_votes):
                percentage = (votes / total_frames_analyzed * 100) if total_frames_analyzed > 0 else 0
                logger.info(f"  Person {idx}: {votes} votes ({percentage:.1f}%)")

            logger.info(f"Person {talking_idx} is talking ({vote_percentage:.1f}% of frames)")
            return talking_idx
        else:
            logger.info("No significant mouth motion detected")
            # Fallback: return first person
            return 0 if num_persons > 0 else None

    def _calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int],
                                bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes.

        Args:
            bbox1: (x1, y1, x2, y2)
            bbox2: (x1, y1, x2, y2)

        Returns:
            Overlap ratio (0.0 to 1.0)
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
