# NIS Protocol Vision Detection Example

This example demonstrates how to use the VisionAgent with YOLO integration for object detection in images and video streams.

## Prerequisites

Before running this example, make sure you have installed the required dependencies:

```
pip install -r requirements.txt
```

This will install OpenCV, the ultralytics YOLO package, and other required libraries.

## Running the Example

The example script supports three modes of operation:

1. **Processing a single image**:
   ```bash
   python run.py --image path/to/image.jpg
   ```

2. **Processing a video file**:
   ```bash
   python run.py --video path/to/video.mp4
   ```

3. **Processing a live camera feed** (default):
   ```bash
   python run.py
   ```
   
   You can specify a different camera index if you have multiple cameras:
   ```bash
   python run.py --camera 1
   ```

### Additional Options

- `--model`: Path to a custom YOLO model weights file (if not specified, uses YOLOv8n by default)
- `--confidence`: Confidence threshold for detections (default: 0.5)
- `--frames`: Maximum number of frames to process for video inputs (default: 1000)

Example with all options:
```bash
python run.py --video path/to/video.mp4 --model path/to/yolov8m.pt --confidence 0.4 --frames 500
```

## Example Usage

```bash
# Process an image
python run.py --image sample_images/street.jpg

# Process a video
python run.py --video sample_videos/traffic.mp4

# Use webcam with higher confidence threshold
python run.py --confidence 0.7
```

## Controls

When in video processing mode:
- Press 'q' to exit the video processing loop

## Understanding the Output

The example will visualize detected objects with bounding boxes, class labels, and confidence scores.

Additionally, the console will display information about the detections when processing images.

## Integration with NIS Protocol

This example demonstrates how the VisionAgent integrates with the broader NIS Protocol framework:

1. It utilizes the NIS Registry for agent management
2. It processes visual inputs using the VisionAgent
3. It demonstrates how the agent's emotional state can be influenced by visual patterns
4. It shows how the processing results are formatted according to the NIS Protocol message structure 