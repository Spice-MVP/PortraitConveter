# Architecture Documentation

## Overview

The Portrait Converter is designed as a modular pipeline that processes landscape podcast videos and converts them to portrait format by intelligently tracking the speaking person.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT VIDEO                              │
│                    (Landscape 16:9)                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SCENE DETECTOR                                │
│  • PySceneDetect with ContentDetector                           │
│  • Splits video based on content changes                        │
│  • Configurable sensitivity threshold                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  Scene 1..N    │
                   └────────┬───────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  ┌──────────┐        ┌──────────┐       ┌──────────┐
  │ Scene 1  │        │ Scene 2  │  ...  │ Scene N  │
  └────┬─────┘        └────┬─────┘       └────┬─────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POSE ESTIMATOR                                │
│  • YOLO-11 Pose model (yolo11n-pose)                            │
│  • Detects persons with confidence >= 0.50                      │
│  • Extracts 17 keypoints including eyes, nose, ears            │
│  • GPU accelerated (CUDA)                                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                  ┌─────────┴──────────┐
                  │                    │
         1 Person │                    │ 2+ Persons
                  ▼                    ▼
        ┌──────────────────┐  ┌──────────────────────┐
        │  SINGLE PERSON   │  │   MULTI-PERSON       │
        │   PROCESSING     │  │    PROCESSING        │
        └────────┬─────────┘  └─────────┬────────────┘
                 │                      │
                 │            ┌─────────▼─────────────────────┐
                 │            │    FACE DETECTOR              │
                 │            │  • RetinaFace                 │
                 │            │  • Confidence >= 0.95         │
                 │            │  • Facial landmarks           │
                 │            └─────────┬─────────────────────┘
                 │                      │
                 │            ┌─────────▼─────────────────────┐
                 │            │  POSE-FACE MATCHING           │
                 │            │  • Verify eyes/nose in bbox   │
                 │            │  • Filter invalid detections  │
                 │            └─────────┬─────────────────────┘
                 │                      │
                 │            ┌─────────▼─────────────────────┐
                 │            │  LIP MOVEMENT ANALYSIS        │
                 │            │  • Extract mouth regions      │
                 │            │  • Calculate variance         │
                 │            │  • Identify talking person    │
                 │            └─────────┬─────────────────────┘
                 │                      │
                 └──────────┬───────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  CROP CALCULATOR                                 │
│  • Collect all eye/nose positions across frames                │
│  • Find bounding box containing all keypoints                  │
│  • Calculate 9:16 crop region                                  │
│  • Position eyes at ~30% from top                              │
│  • STATIC crop (no frame-by-frame tracking)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VIDEO CROPPING                                  │
│  • Apply calculated crop to all frames                         │
│  • Resize to output resolution (1080x1920)                     │
│  • Save portrait scene video                                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                  ┌────────────────┐
                  │ Portrait Scenes │
                  │   1, 2, ... N   │
                  └────────┬────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  VIDEO STITCHER                                  │
│  • Concatenate all portrait scenes                             │
│  • Preserve frame rate and resolution                          │
│  • Use FFmpeg for optimal quality                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AUDIO PROCESSOR                                 │
│  • Extract audio from original video                           │
│  • Copy audio stream to portrait video                         │
│  • FFmpeg AAC encoding                                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                      OUTPUT VIDEO                                │
│                    (Portrait 9:16)                               │
│              With original audio preserved                       │
└─────────────────────────────────────────────────────────────────┘
```

## Module Breakdown

### 1. Scene Detector (`src/processors/scene_detector.py`)
- **Purpose**: Split video into scenes based on content changes
- **Algorithm**: PySceneDetect with ContentDetector
- **Input**: Landscape video file
- **Output**: List of scene video files
- **Key Parameters**:
  - `SCENE_THRESHOLD`: Sensitivity (default: 27.0)
  - `MIN_SCENE_LENGTH`: Minimum frames per scene (default: 15)

### 2. Pose Estimator (`src/models/pose_estimator.py`)
- **Purpose**: Detect persons and extract pose keypoints
- **Model**: YOLO-11 Pose (nano variant)
- **Input**: Individual video frames
- **Output**: List of person detections with:
  - Bounding box coordinates
  - 17 keypoint positions
  - Confidence scores
  - Eye and nose positions
- **Key Features**:
  - GPU acceleration via PyTorch
  - Confidence filtering (>= 0.80)
  - Keypoint confidence validation

### 3. Face Detector (`src/models/face_detector.py`)
- **Purpose**: Detect faces and analyze lip movement
- **Model**: RetinaFace
- **Components**:
  - **Face Detection**: High-precision face localization
  - **Landmark Extraction**: Eyes, nose, mouth corners
  - **Lip Movement Analysis**: Variance calculation on mouth regions
- **Input**: Video frames
- **Output**:
  - Face bounding boxes
  - Facial landmarks
  - Lip movement scores
- **Key Features**:
  - Pose-face matching algorithm
  - Temporal lip movement analysis
  - Talking person identification

### 4. Crop Calculator (`src/processors/crop_calculator.py`)
- **Purpose**: Calculate optimal static crop region
- **Algorithm**:
  1. Collect all eye/nose positions across frames
  2. Find bounding box containing all positions
  3. Calculate 9:16 crop region
  4. Position eyes at 30% from top
  5. Ensure all keypoints remain in frame
- **Input**: Person detections for all frames
- **Output**: (x, y, width, height) crop coordinates
- **Key Features**:
  - Static crop (no tracking jitter)
  - Automatic padding adjustment
  - Aspect ratio preservation

### 5. Video Stitcher (`src/processors/video_stitcher.py`)
- **Purpose**: Combine portrait scenes and add audio
- **Methods**:
  - **OpenCV stitching**: Frame-by-frame concatenation
  - **FFmpeg stitching**: Fast concat demuxer (preferred)
- **Input**: List of portrait scene videos
- **Output**: Final portrait video with audio
- **Key Features**:
  - Audio stream copying (lossless)
  - Fallback mechanisms
  - AAC audio encoding

### 6. Visualizer (`src/utils/visualizer.py`)
- **Purpose**: Debug visualizations
- **Features**:
  - Draw pose keypoints and bounding boxes
  - Draw face landmarks
  - Highlight crop regions
  - Annotate detection methods
  - Save debug frames at intervals
- **Output**: Annotated debug images

### 7. Pipeline Orchestrator (`src/pipeline.py`)
- **Purpose**: Main workflow coordination
- **Responsibilities**:
  - Initialize all components
  - Process each scene sequentially
  - Handle single vs multi-person logic
  - Manage temp directories
  - Error handling and logging
- **Key Methods**:
  - `convert()`: Main entry point
  - `process_scene()`: Per-scene processing
  - `_process_single_person_scene()`: 1-person logic
  - `_process_multi_person_scene()`: 2+ person logic
  - `_fallback_to_best_person()`: Error recovery

## Data Flow

### Single Person Scene
```
Frame → YOLO → Person Detection → Eye/Nose Extraction
                                          ↓
                                  Collect All Frames
                                          ↓
                                  Calculate Crop
                                          ↓
                                  Apply Crop
                                          ↓
                                  Portrait Video
```

### Multi-Person Scene
```
Frame → YOLO → Person Detections
                      ↓
                RetinaFace → Face Detections
                      ↓
                Match Poses with Faces
                      ↓
              Analyze Lip Movement
                      ↓
              Select Talking Person
                      ↓
              Collect Keypoints
                      ↓
              Calculate Crop
                      ↓
              Apply Crop
                      ↓
              Portrait Video
```

## Key Design Decisions

### 1. Static vs Dynamic Cropping
**Decision**: Use static crop per scene

**Rationale**:
- Eliminates jitter and shakiness
- Simpler implementation
- More predictable results
- Faster processing (one calculation per scene)

**Trade-off**: Less adaptive to subject movement

### 2. Scene-Based Processing
**Decision**: Split into scenes first

**Rationale**:
- Different people may speak in different scenes
- Allows per-scene crop optimization
- Better handles camera angle changes
- Enables parallel processing (future)

### 3. YOLO + RetinaFace Combination
**Decision**: Use both models

**Rationale**:
- YOLO: Fast pose detection with GPU support
- RetinaFace: High-precision face detection
- Complementary strengths
- Validation via pose-face matching

### 4. Lip Movement Variance
**Decision**: Use pixel variance as proxy for talking

**Rationale**:
- Simple and fast
- No audio analysis needed
- Works with muted videos
- Sufficient for podcast scenarios

**Limitation**: May struggle with side profiles or heavy compression

### 5. Threshold-Based Filtering
**Decision**: Use high confidence thresholds (0.80, 0.95)

**Rationale**:
- Reduces false positives
- Ensures reliable keypoint extraction
- Better for professional content

**Trade-off**: May miss some valid detections

## Performance Optimization

### GPU Acceleration
- YOLO-11 runs on CUDA (if available)
- Batch processing for pose estimation
- RetinaFace on CPU (library limitation)

### Memory Management
- Frame-by-frame processing (no full video in RAM)
- Temporary scene files for intermediate results
- Configurable output resolution

### Processing Speed
- Scene detection: Fast (content analysis)
- Pose estimation: Medium (GPU accelerated)
- Face detection: Slow (CPU only)
- Lip movement: Slow (multiple frame reads)
- Cropping: Fast (vectorized operations)

## Error Handling

### Fallback Strategies
1. **No persons detected**: Skip scene, log warning
2. **No faces detected**: Use highest confidence person
3. **No lip movement**: Use first matched person
4. **Invalid crop**: Expand with padding

### Validation
- Keypoint confidence checking
- Pose-face bbox verification
- Crop bounds validation
- Audio stream availability

## Configuration

All settings in `src/utils/config.py`:
- Detection thresholds
- Output dimensions
- Scene detection parameters
- Debug settings
- GPU/CPU selection

## Future Enhancements

1. **Dynamic Tracking**: Smooth crop transitions within scenes
2. **Audio Analysis**: Use speech detection for talking person
3. **Multi-threading**: Parallel scene processing
4. **Model Optimization**: Quantization for faster inference
5. **Real-time Preview**: Live preview during processing
6. **Batch Mode**: Process multiple videos
7. **API Server**: REST API for cloud deployment
