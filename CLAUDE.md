# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PortraitConverter is a Python video processing pipeline that converts landscape podcast videos (16:9) to portrait format (9:16) by intelligently tracking and cropping to the speaking person. It uses YOLO-11 for pose estimation, RetinaFace for face detection, and lip movement analysis to identify who is talking in multi-person scenes.

## Development Commands

### Setup & Installation
```bash
# Initial setup (creates venv and installs dependencies)
./setup_venv.sh

# Verify installation
source venv/bin/activate
python test_installation.py
deactivate
```

### Running the Converter
```bash
# Using convenience script (auto-activates venv)
./run.sh --input video.mp4 --output portrait.mp4

# With debug mode (saves annotated frames to debug_frames/)
./run.sh --input video.mp4 --output portrait.mp4 --debug

# Manual activation
source venv/bin/activate
python main.py --input video.mp4 --output portrait.mp4
deactivate
```

### Common Options
- `--debug`: Enable debug mode (saves detection frames)
- `--pose-threshold 0.80`: YOLO pose detection confidence (0.0-1.0)
- `--face-threshold 0.95`: RetinaFace detection confidence (0.0-1.0)
- `--scene-threshold 27.0`: Scene detection sensitivity (higher = fewer scenes)
- `--output-width 1080` / `--output-height 1920`: Output dimensions

## Architecture Overview

### Processing Pipeline Flow

The conversion happens in 4 main stages orchestrated by `PortraitConverter` in [src/pipeline.py](src/pipeline.py):

1. **Scene Detection** → 2. **Person/Face Detection** → 3. **Crop Calculation** → 4. **Stitching + Audio**

### Key Architectural Decisions

**Static Crop Per Scene**: Unlike tracking-based systems, this uses a single static crop region calculated for each scene. All eye/nose positions across frames are collected, a bounding box is computed to contain them all, and a 9:16 crop region is calculated with eyes positioned ~30% from top. This eliminates jitter but requires persons to stay relatively stationary within scenes.

**Scene-Based Processing**: Video is split into scenes first using PySceneDetect's ContentDetector. Each scene is processed independently, allowing different speakers to be focused on in different scenes. Scenes are later stitched back together.

**Single vs Multi-Person Logic**:
- **1 person detected**: Immediately use that person's keypoints for crop calculation
- **2+ persons detected**: Run RetinaFace → match poses to faces → analyze lip movement → select talking person

**Pose-Face Matching**: When multiple persons are detected, the system validates that YOLO pose keypoints (eyes, nose) fall within RetinaFace bounding boxes. This filters out false detections and ensures reliable person identification.

**Talking Person Detection**: For multi-person scenes, the system uses frame-to-frame motion detection with a voting system:
- Extracts lower third of each face (mouth region)
- Compares consecutive frames using `cv2.absdiff()` to detect pixel changes
- Each frame "votes" for the person with highest mouth motion (>2% threshold)
- Person with most votes across the scene is selected as talking person
- More robust than variance-based approaches as it focuses on inter-frame changes rather than absolute pixel values

### Module Responsibilities

**`src/pipeline.py` (PortraitConverter)**: Main orchestrator. Coordinates scene processing, dispatches to single vs multi-person workflows, handles fallback strategies when detections fail.

**`src/processors/scene_detector.py` (SceneDetector)**: Uses PySceneDetect ContentDetector to split video into scenes based on content changes. Configurable via `Config.SCENE_THRESHOLD` and `Config.MIN_SCENE_LENGTH`.

**`src/models/pose_estimator.py` (PoseEstimator)**: Wraps YOLO-11 Pose (nano variant). Detects persons and extracts 17 keypoints including eyes, nose, ears. GPU-accelerated via PyTorch. Filters detections by `Config.POSE_THRESHOLD` (default 0.80).

**`src/models/face_detector.py` (FaceDetector)**: Wraps RetinaFace for high-precision face detection. Provides:
- Face bounding boxes and landmarks
- Pose-face matching algorithm (`is_pose_in_face()`)
- Frame-to-frame motion detection (`find_talking_person()`)
- Mouth region extraction (`extract_mouth_region_from_face()`)
- Motion calculation using pixel differences (`calculate_mouth_motion()`)

**`src/processors/crop_calculator.py` (CropCalculator)**: Calculates and applies static crop regions. Key methods:
- `calculate_static_crop()`: Collects all eye/nose positions, finds bounding box, calculates 9:16 crop with configurable padding
- `calculate_center_crop()`: Fallback for scenes with no person detection - centers a 9:16 crop on the frame
- `apply_crop()`: Uses OpenCV to crop and resize frames

**`src/processors/video_stitcher.py` (VideoStitcher)**: Concatenates portrait scene videos using FFmpeg's concat demuxer (fast, no re-encoding). Adds audio from original video using FFmpeg audio stream copying.

**`src/utils/config.py` (Config)**: Global configuration class. All thresholds, dimensions, and settings are centralized here. GPU/CPU device selection via `torch.cuda.is_available()`.

**`src/utils/visualizer.py` (Visualizer)**: Debug visualization system. Draws pose keypoints, face landmarks, crop regions on frames. Saves annotated frames at intervals when debug mode is enabled.

**`src/utils/logger.py`**: Logging configuration.

### Data Flow: Multi-Person Scene Example

```
Frame → PoseEstimator.detect_poses() → List[Dict] with bbox, keypoints, confidence
          ↓
Frame → FaceDetector.detect_faces() → List[Dict] with bbox, landmarks
          ↓
FaceDetector.is_pose_in_face() → Match poses to faces (verify eyes/nose in bbox)
          ↓
FaceDetector.find_talking_person() → Frame-to-frame motion detection with voting
          ↓
  For each sampled frame:
    - Extract mouth region (lower 1/3 of face)
    - Calculate cv2.absdiff() with previous frame
    - Vote for person with highest motion (>2% threshold)
    - Tally votes across all frames
          ↓
_extract_target_person_across_frames() → Track target using IoU matching
          ↓
CropCalculator.calculate_static_crop() → Collect keypoints, compute bbox, calculate 9:16 crop
          ↓
CropCalculator.apply_crop() → Apply crop to all frames, resize to output dimensions
```

### Fallback Strategies

The pipeline is resilient to detection failures:
1. **No persons detected**: Use center crop fallback (maintains 9:16 aspect ratio, centered on frame)
2. **No faces detected** (multi-person): Use person with highest YOLO confidence
3. **No lip movement detected**: Use first person with confirmed face match
4. **Invalid crop region**: Apply padding expansion to ensure valid crop

The center crop fallback is implemented in `_process_no_person_scene()` and ensures that scenes without detectable persons are still included in the output video rather than being skipped. Other fallbacks are implemented in `_fallback_to_best_person()` in [src/pipeline.py](src/pipeline.py).

### Temporary Files & Directories

- `temp_work/scenes/`: Scene video files after splitting
- `temp_work/portraits/`: Portrait videos for each scene (before stitching)
- `debug_frames/`: Debug visualization frames (if `--debug` enabled)
- `output/`: User-specified output location (not in repo)

All temp directories are created in `convert()` method and can be manually cleaned up.

## Configuration & Thresholds

All configuration is in [src/utils/config.py](src/utils/config.py). Key settings:

**Detection Thresholds**:
- `POSE_THRESHOLD = 0.50` (default in code, 0.80 in CLI): YOLO confidence filter
- `FACE_THRESHOLD = 0.95`: RetinaFace confidence filter
- Higher = fewer false positives, may miss valid detections
- Lower = more detections, more false positives

**Scene Detection**:
- `SCENE_THRESHOLD = 27.0`: Content change sensitivity (higher = fewer scenes)
- `MIN_SCENE_LENGTH = 15`: Minimum frames per scene

**Crop Positioning**:
- `CROP_PADDING_TOP = 0.3`: 30% padding above eyes
- `CROP_PADDING_BOTTOM = 0.5`: 50% padding below nose
- `CROP_PADDING_SIDES = 0.15`: 15% padding on sides
- These control headroom and framing in the portrait crop

**Motion Detection**:
- Motion threshold: `0.02` (2% pixel change) - hardcoded in `find_talking_person()`
- Sample rate: Every 3rd frame for efficiency
- Voting system: Frame-by-frame votes tallied across scene

## Programmatic Usage

```python
from src.pipeline import PortraitConverter
from src.utils.config import Config

# Configure thresholds
Config.set_thresholds(pose_threshold=0.80, face_threshold=0.95)
Config.SCENE_THRESHOLD = 27.0
Config.OUTPUT_WIDTH = 1080
Config.OUTPUT_HEIGHT = 1920

# Create converter with debug mode
converter = PortraitConverter(debug_mode=True)

# Convert (returns True if successful)
success = converter.convert("input.mp4", "output.mp4")
```

See [example_usage.py](example_usage.py) for more examples including custom resolutions and error handling.

## Performance Considerations

**GPU Acceleration**: YOLO-11 runs on CUDA if available (detected via `torch.cuda.is_available()`). RetinaFace library limitation forces CPU execution for face detection, which slows multi-person scenes.

**Memory**: Frame-by-frame processing keeps memory usage low (~2-4GB RAM, ~2-4GB VRAM). No full video loading. Temporary scene files are written to disk.

**Speed Bottlenecks**:
- Scene detection: Fast
- YOLO pose estimation: Medium (GPU accelerated)
- RetinaFace detection: Slow (CPU only)
- Lip movement analysis: Slow (requires multiple frame reads)
- Cropping/stitching: Fast

**Optimization Tips**: Lower output resolution for faster processing. Multi-person scenes are significantly slower due to face detection and lip analysis.

## Known Limitations

1. **Static Crop**: No dynamic tracking or smoothing. Persons must stay relatively stationary within each scene.
2. **Motion Detection Limitations**:
   - Works best with clear frontal faces
   - Head movement while talking may be detected as motion (by design - indicates engagement)
   - Extreme head turns or side profiles reduce accuracy
   - Requires at least 30% bbox overlap to track faces across frames
3. **Scene Detection**: May over/under-segment with gradual lighting changes or frequent cuts.
4. **Multi-Person Scale**: Best with 2-3 people. May struggle with 5+ people or overlapping faces.

## Dependencies

Core dependencies (see [requirements.txt](requirements.txt)):
- `opencv-python` + `opencv-contrib-python`: Video I/O, frame processing
- `ultralytics`: YOLO-11 pose estimation
- `torch` + `torchvision`: PyTorch for GPU acceleration
- `retina-face`: Face detection and landmarks
- `scenedetect`: Scene splitting
- `numpy`, `scipy`: Numerical operations
- `tqdm`: Progress bars

External dependency:
- **FFmpeg**: Required for audio handling. Must be installed separately (`brew install ffmpeg` on macOS).

## Testing

Run installation test to verify all dependencies:
```bash
python test_installation.py
```

This checks:
- All Python packages can be imported
- GPU availability (informational)
- FFmpeg installation
- Project file structure
- Module imports

Currently no unit tests exist. When adding tests, focus on:
- Crop calculation edge cases (person at frame edge, multiple persons)
- Pose-face matching algorithm accuracy
- Scene detection sensitivity
- IoU tracking across frames
