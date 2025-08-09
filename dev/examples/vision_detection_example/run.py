#!/usr/bin/env python3
"""
NIS Protocol Vision Agent Example

This example demonstrates how to use the VisionAgent with YOLO integration
for object detection in images and video streams.
"""

import os
import sys
import time
import cv2
import numpy as np
import argparse
from typing import Dict, Any, Optional

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.agents.perception.vision_agent import VisionAgent
from src.emotion.emotional_state import EmotionalState
from src.core.registry import NISRegistry

def display_detections(image: np.ndarray, detections: list, title: str = "Object Detection") -> np.ndarray:
    """
    Display detections on an image with bounding boxes and labels.
    
    Args:
        image: Input image
        detections: List of detections from the vision agent
        title: Window title
        
    Returns:
        Image with detections drawn on it
    """
    # Make a copy of the image to draw on
    display_img = image.copy()
    
    # Get image dimensions
    height, width = display_img.shape[:2]
    
    # Draw each detection
    for detection in detections:
        if "bbox" in detection and "class" in detection and "confidence" in detection:
            # Get detection info
            bbox = detection["bbox"]
            class_name = detection["class"]
            confidence = detection["confidence"]
            
            # Ensure bbox coordinates are valid
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))
            
            # Generate a color based on class name (consistent for same classes)
            color_hash = sum(ord(c) for c in class_name) % 255
            color = (color_hash, 255 - color_hash, (color_hash * 123) % 255)
            
            # Draw bounding box
            cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                display_img, 
                (x1, y1 - label_height - 10), 
                (x1 + label_width + 10, y1), 
                color, 
                -1
            )
            
            # Draw label text
            cv2.putText(
                display_img, 
                label, 
                (x1 + 5, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                1, 
                cv2.LINE_AA
            )
            
            # Draw dominant color if available
            if "dominant_color" in detection:
                dom_color = detection["dominant_color"]
                cv2.rectangle(
                    display_img, 
                    (x2 - 20, y1), 
                    (x2, y1 + 20), 
                    (int(dom_color[0]), int(dom_color[1]), int(dom_color[2])), 
                    -1
                )
    
    return display_img

def process_image(vision_agent: VisionAgent, image_path: str, display: bool = True) -> Dict[str, Any]:
    """
    Process a single image with the vision agent.
    
    Args:
        vision_agent: Initialized vision agent
        image_path: Path to the image file
        display: Whether to display the results
        
    Returns:
        Agent's response
    """
    # Process the image
    response = vision_agent.process({
        "image_data": image_path
    })
    
    if display and os.path.exists(image_path):
        # Load the image
        image = cv2.imread(image_path)
        
        if image is not None and "detections" in response["payload"]:
            # Draw detections on the image
            display_img = display_detections(image, response["payload"]["detections"])
            
            # Show the image
            cv2.imshow("NIS Vision Agent - Object Detection", display_img)
            cv2.waitKey(0)
    
    return response

def process_video(vision_agent: VisionAgent, video_source: int, max_frames: int = 100) -> None:
    """
    Process a video stream with the vision agent.
    
    Args:
        vision_agent: Initialized vision agent
        video_source: Camera index or video file path
        max_frames: Maximum number of frames to process
    """
    # Start video capture
    response = vision_agent.process({
        "video_source": video_source
    })
    
    # Setup window
    window_name = "NIS Vision Agent - Video Stream"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    frame_count = 0
    last_time = time.time()
    fps = 0
    
    # Process frames in a loop
    while frame_count < max_frames:
        start_time = time.time()
        
        # Process current frame
        response = vision_agent.process({
            "video_source": video_source
        })
        
        # Calculate FPS
        current_time = time.time()
        if current_time - last_time > 0.5:  # Update FPS every 0.5 seconds
            fps = 1.0 / (current_time - start_time)
            last_time = current_time
        
        # Check for valid detections
        if "payload" in response and "detections" in response["payload"]:
            detections = response["payload"]["detections"]
            
            # Get a frame from the video capture (the agent already has it internally)
            if vision_agent.is_streaming and vision_agent.video_capture is not None:
                ret, frame = vision_agent.video_capture.read()
                
                if ret:
                    # Draw detections on the frame
                    display_img = display_detections(frame, detections)
                    
                    # Add FPS counter
                    cv2.putText(
                        display_img,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                    
                    # Show the frame
                    cv2.imshow(window_name, display_img)
                    
                    # Break loop on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # End of video or error
                    break
            else:
                # No valid frame
                break
        
        frame_count += 1
    
    # Stop streaming
    vision_agent.process({"stop_stream": True})
    cv2.destroyAllWindows()

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="NIS Vision Agent Example")
    parser.add_argument("--image", type=str, help="Path to an image file")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for video input")
    parser.add_argument("--video", type=str, help="Path to a video file")
    parser.add_argument("--model", type=str, help="Path to a YOLO model weights file")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detections")
    parser.add_argument("--frames", type=int, default=1000, help="Maximum number of frames to process")
    args = parser.parse_args()
    
    # Initialize the NIS Registry
    registry = NISRegistry()
    
    # Initialize the vision agent
    vision_agent = VisionAgent(
        agent_id="vision_demo",
        description="Vision agent for object detection demo",
        emotional_state=EmotionalState(),
        yolo_model_path=args.model,
        confidence_threshold=args.confidence
    )
    
    # Process image if specified
    if args.image:
        if os.path.exists(args.image):
            print(f"Processing image: {args.image}")
            response = process_image(vision_agent, args.image)
            
            if "payload" in response and "detections" in response["payload"]:
                print(f"Found {len(response['payload']['detections'])} objects")
                for detection in response["payload"]["detections"]:
                    if "class" in detection and "confidence" in detection:
                        print(f"  - {detection['class']} ({detection['confidence']:.2f})")
            else:
                print("No detections found")
        else:
            print(f"Error: Image file not found: {args.image}")
    
    # Process video if specified
    elif args.video:
        if os.path.exists(args.video):
            print(f"Processing video: {args.video}")
            process_video(vision_agent, args.video, args.frames)
        else:
            print(f"Error: Video file not found: {args.video}")
    
    # Otherwise use camera
    else:
        print(f"Starting camera stream (index {args.camera})")
        process_video(vision_agent, args.camera, args.frames)

if __name__ == "__main__":
    main() 