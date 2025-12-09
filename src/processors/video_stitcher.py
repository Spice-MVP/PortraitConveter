"""Video stitching and audio management."""

import cv2
import os
import subprocess
from typing import List

from ..utils.logger import logger


class VideoStitcher:
    """Stitch portrait videos and add audio."""

    @staticmethod
    def stitch_videos(video_paths: List[str], output_path: str) -> bool:
        """Stitch multiple videos together.

        Args:
            video_paths: List of video file paths to stitch
            output_path: Output video path

        Returns:
            True if successful
        """
        if not video_paths:
            logger.error("No videos to stitch")
            return False

        logger.info(f"Stitching {len(video_paths)} videos")

        # Get properties from first video
        cap = cv2.VideoCapture(video_paths[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_frames = 0
        for i, video_path in enumerate(video_paths):
            logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")

            cap = cv2.VideoCapture(video_path)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                out.write(frame)
                total_frames += 1

            cap.release()

        out.release()

        logger.info(f"Stitched {total_frames} total frames to {output_path}")

        return True

    @staticmethod
    def add_audio(video_path: str, audio_source: str, output_path: str) -> bool:
        """Add audio from source video to output video.

        Args:
            video_path: Video file (without audio)
            audio_source: Source video with audio
            output_path: Output video path with audio

        Returns:
            True if successful
        """
        logger.info(f"Adding audio from {audio_source} to {video_path}")

        try:
            # Use ffmpeg to copy audio stream
            cmd = [
                'ffmpeg',
                '-i', video_path,      # Input video
                '-i', audio_source,    # Input audio source
                '-c:v', 'copy',        # Copy video codec
                '-c:a', 'aac',         # Encode audio as AAC
                '-map', '0:v:0',       # Use video from first input
                '-map', '1:a:0',       # Use audio from second input
                '-shortest',           # Finish when shortest stream ends
                '-y',                  # Overwrite output
                output_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode == 0:
                logger.info(f"Successfully added audio to {output_path}")
                return True
            else:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False

        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg to add audio.")
            return False
        except Exception as e:
            logger.error(f"Error adding audio: {e}")
            return False

    @staticmethod
    def create_concat_file(video_paths: List[str], concat_file: str):
        """Create FFmpeg concat file for stitching.

        Args:
            video_paths: List of video paths
            concat_file: Output concat file path
        """
        with open(concat_file, 'w') as f:
            for video_path in video_paths:
                # Use absolute path
                abs_path = os.path.abspath(video_path)
                f.write(f"file '{abs_path}'\n")

    @staticmethod
    def stitch_with_ffmpeg(video_paths: List[str], output_path: str) -> bool:
        """Stitch videos using ffmpeg (better quality).

        Args:
            video_paths: List of video paths
            output_path: Output path

        Returns:
            True if successful
        """
        if not video_paths:
            return False

        logger.info(f"Stitching {len(video_paths)} videos with ffmpeg")

        try:
            # Create concat file
            concat_file = output_path + '.concat.txt'
            VideoStitcher.create_concat_file(video_paths, concat_file)

            # Use ffmpeg to concatenate
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', concat_file,
                '-c', 'copy',
                '-y',
                output_path
            ]

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Clean up concat file
            if os.path.exists(concat_file):
                os.remove(concat_file)

            if result.returncode == 0:
                logger.info(f"Successfully stitched videos to {output_path}")
                return True
            else:
                logger.warning(f"FFmpeg concat failed: {result.stderr}")
                logger.info("Falling back to OpenCV stitching")
                return VideoStitcher.stitch_videos(video_paths, output_path)

        except FileNotFoundError:
            logger.warning("ffmpeg not found. Using OpenCV for stitching")
            return VideoStitcher.stitch_videos(video_paths, output_path)
        except Exception as e:
            logger.error(f"Error stitching with ffmpeg: {e}")
            return False
