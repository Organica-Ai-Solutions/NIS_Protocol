"""
Vision Agent

Processes visual inputs such as images, video streams, or UI interactions.
Analogous to the visual cortex in the brain. Implements YOLO-based object detection
and OpenCV for image processing.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import time
import numpy as np
import cv2
import os

from src.core.registry import NISAgent, NISLayer, NISRegistry
from src.emotion.emotional_state import EmotionalState, EmotionalDimension


class YOLOModel:
    """
    Wrapper class for YOLO object detection model.
    Supports both YOLOv5/YOLOv8 via ultralytics and OpenCV's DNN module for older YOLO models.
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize the YOLO model.
        
        Args:
            model_path: Path to the YOLO model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.use_ultralytics = False
        
        # Try to load the model
        self._load_model()
    
    def _load_model(self):
        """Load the appropriate YOLO model."""
        try:
            # First try to load using ultralytics (YOLOv5/v8)
            import torch
            from ultralytics import YOLO
            
            # Use default YOLOv8n if no model specified
            if self.model_path is None or not os.path.exists(self.model_path):
                self.model = YOLO('yolov8n.pt')
            else:
                self.model = YOLO(self.model_path)
            
            self.use_ultralytics = True
            self.class_names = self.model.names
            
        except (ImportError, ModuleNotFoundError):
            # Fallback to OpenCV DNN implementation
            print("Ultralytics not found, falling back to OpenCV DNN implementation")
            self.use_ultralytics = False
            
            # Default to COCO classes if using OpenCV DNN
            self.class_names = self._get_coco_classes()
            
            if self.model_path is None or not os.path.exists(self.model_path):
                # Try to download a default model
                self._download_default_model()
            
            if os.path.exists(self.model_path):
                self.model = cv2.dnn.readNetFromDarknet(
                    self.model_path.replace('.weights', '.cfg'),
                    self.model_path
                )
            else:
                print("Warning: No YOLO model available")
    
    def _download_default_model(self):
        """Download a default YOLO model if none is provided."""
        try:
            import gdown
            
            # Set default model path
            self.model_path = "yolov4-tiny.weights"
            cfg_path = "yolov4-tiny.cfg"
            
            # Download weights and config if not exists
            if not os.path.exists(self.model_path):
                print("Downloading default YOLOv4-tiny model...")
                # YOLOv4-tiny weights
                gdown.download(
                    "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights",
                    self.model_path,
                    quiet=False
                )
            
            if not os.path.exists(cfg_path):
                # YOLOv4-tiny config
                gdown.download(
                    "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
                    cfg_path,
                    quiet=False
                )
        except ImportError:
            print("Could not download default model: gdown not installed")
    
    def _get_coco_classes(self) -> List[str]:
        """Get the list of COCO classes."""
        return [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image.
        
        Args:
            image: Image as numpy array (BGR format)
            
        Returns:
            List of detections with class, confidence, and bounding box
        """
        if self.model is None:
            return []
        
        if self.use_ultralytics:
            return self._detect_ultralytics(image)
        else:
            return self._detect_opencv(image)
    
    def _detect_ultralytics(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects using the ultralytics YOLO model."""
        results = self.model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get prediction info
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                if confidence >= self.confidence_threshold:
                    detections.append({
                        "class": self.class_names[class_id],
                        "confidence": confidence,
                        "bbox": [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        return detections
    
    def _detect_opencv(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects using OpenCV DNN and YOLO model."""
        height, width = image.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            image, 1/255.0, (416, 416), swapRB=True, crop=False
        )
        
        # Set input and get output
        self.model.setInput(blob)
        
        # Get output layer names
        out_layer_names = self.model.getUnconnectedOutLayersNames()
        outputs = self.model.forward(out_layer_names)
        
        # Process outputs
        detections = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = float(scores[class_id])
                
                if confidence > self.confidence_threshold:
                    # YOLO returns center, width, height
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    detections.append({
                        "class": self.class_names[class_id],
                        "confidence": confidence,
                        "bbox": [x, y, x + w, y + h]
                    })
        
        return detections


class VisionAgent(NISAgent):
    """
    Agent that processes visual inputs (images, video streams, or UI interactions)
    
    The Vision Agent is responsible for:
    - Processing raw visual inputs with OpenCV
    - Detecting objects using YOLO models
    - Translating visual information into structured data
    - Identifying anomalies in visual patterns
    """
    
    def __init__(
        self,
        agent_id: str = "vision",
        description: str = "Processes visual inputs and identifies objects using YOLO",
        emotional_state: Optional[EmotionalState] = None,
        yolo_model_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize a new Vision Agent.
        
        Args:
            agent_id: Unique identifier for this agent
            description: Human-readable description of the agent's role
            emotional_state: Optional pre-configured emotional state
            yolo_model_path: Path to YOLO model weights file
            confidence_threshold: Minimum confidence for object detection
        """
        super().__init__(agent_id, NISLayer.PERCEPTION, description)
        self.emotional_state = emotional_state or EmotionalState()
        self.detection_history = []
        
        # Initialize YOLO model
        self.object_detector = YOLOModel(
            model_path=yolo_model_path,
            confidence_threshold=confidence_threshold
        )
        
        # Initialize video capture for streaming sources
        self.video_capture = None
        self.is_streaming = False
    
    def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a visual input and extract features.
        
        Args:
            message: Message containing visual data
                'image_data': Image as numpy array or path to image file
                'video_source': Camera index or video file path
                'ui_state': UI element state (optional)
                'text_data': Visible text (optional)
                'stop_stream': Boolean to stop an active video stream
        
        Returns:
            Processed features and detected objects
        """
        start_time = self._start_processing_timer()
        
        if not self._validate_message(message):
            return self._create_response(
                "error",
                {"error": "Invalid message format or missing required fields"}
            )
        
        # Handle stream stop if requested
        if message.get("stop_stream", False) and self.is_streaming:
            self._stop_streaming()
            return self._create_response(
                "success",
                {"message": "Video stream stopped"}
            )
        
        # Process based on input type
        features = {}
        detections = []
        
        # Process image data if present
        if "image_data" in message:
            image = self._load_image(message["image_data"])
            if image is not None:
                image_features, image_detections = self._process_image(image)
                features.update(image_features)
                detections.extend(image_detections)
        
        # Process video source if present
        if "video_source" in message:
            self._setup_video_stream(message["video_source"])
            if self.is_streaming:
                frame = self._get_video_frame()
                if frame is not None:
                    video_features, video_detections = self._process_image(frame)
                    features.update(video_features)
                    detections.extend(video_detections)
        
        # Process UI state if present
        if "ui_state" in message:
            ui_features, ui_detections = self._process_ui(message["ui_state"])
            features.update(ui_features)
            detections.extend(ui_detections)
        
        # Process visible text if present
        if "text_data" in message:
            text_features, text_detections = self._process_text(message["text_data"])
            features.update(text_features)
            detections.extend(text_detections)
        
        # Update emotional state based on detections
        unusual_patterns = self._detect_unusual_patterns(detections)
        if unusual_patterns:
            self.emotional_state.update(EmotionalDimension.SUSPICION.value, 0.7)
            self.emotional_state.update(EmotionalDimension.NOVELTY.value, 0.8)
        
        # Track detection history (limited to most recent 100)
        self.detection_history.append({
            "timestamp": time.time(),
            "detections": detections
        })
        if len(self.detection_history) > 100:
            self.detection_history.pop(0)
        
        self._end_processing_timer(start_time)
        
        return self._create_response(
            "success",
            {
                "features": features,
                "detections": detections
            },
            metadata={
                "is_streaming": self.is_streaming
            },
            emotional_state=self.emotional_state.get_state()
        )
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """
        Validate incoming message format.
        
        Args:
            message: The message to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(message, dict):
            return False
            
        # Special case for stop_stream command
        if message.get("stop_stream", False):
            return True
            
        # Must contain at least one of these data fields
        required_fields = ["image_data", "video_source", "ui_state", "text_data"]
        return any(field in message for field in required_fields)
    
    def _load_image(self, image_data: Union[str, np.ndarray, bytes]) -> Optional[np.ndarray]:
        """
        Load image from various input formats.
        
        Args:
            image_data: Can be:
                - Path to image file (str)
                - Image as numpy array
                - Image as bytes
                
        Returns:
            Image as numpy array or None if loading failed
        """
        try:
            if isinstance(image_data, str):
                # Path to image file
                if os.path.exists(image_data):
                    return cv2.imread(image_data)
            elif isinstance(image_data, np.ndarray):
                # Already a numpy array
                return image_data
            elif isinstance(image_data, bytes):
                # Bytes data
                nparr = np.frombuffer(image_data, np.uint8)
                return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return None
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def _setup_video_stream(self, video_source: Union[int, str]) -> None:
        """
        Setup video capture for streaming.
        
        Args:
            video_source: Camera index (int) or video file path (str)
        """
        # Stop existing stream if any
        if self.video_capture is not None:
            self._stop_streaming()
        
        try:
            # Initialize video capture
            self.video_capture = cv2.VideoCapture(video_source)
            if self.video_capture.isOpened():
                self.is_streaming = True
            else:
                print(f"Error: Could not open video source {video_source}")
                self.is_streaming = False
        except Exception as e:
            print(f"Error setting up video stream: {e}")
            self.is_streaming = False
    
    def _stop_streaming(self) -> None:
        """Stop the current video stream."""
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.is_streaming = False
    
    def _get_video_frame(self) -> Optional[np.ndarray]:
        """
        Get a frame from the current video stream.
        
        Returns:
            Frame as numpy array or None if no frame available
        """
        if not self.is_streaming or self.video_capture is None:
            return None
        
        ret, frame = self.video_capture.read()
        if ret:
            return frame
        else:
            # End of stream or error
            self._stop_streaming()
            return None
    
    def _process_image(self, image: np.ndarray) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process image data to extract features and detections.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Tuple of (features, detections)
        """
        if image is None:
            return {}, []
        
        height, width = image.shape[:2]
        
        # Extract basic image features
        features = {
            "image_processed": True,
            "resolution": f"{width}x{height}",
            "channels": image.shape[2] if len(image.shape) > 2 else 1,
            "mean_pixel_value": float(np.mean(image)),
            "std_pixel_value": float(np.std(image))
        }
        
        # Use YOLO for object detection
        detections = self.object_detector.detect(image)
        
        # Apply further OpenCV processing (example: edge detection)
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            features["has_edges"] = np.count_nonzero(edges) > (width * height * 0.01)
            features["edge_density"] = float(np.count_nonzero(edges) / (width * height))
        except Exception as e:
            print(f"Error processing image edges: {e}")
        
        # Enhance detections with color information
        for detection in detections:
            if "bbox" in detection:
                x1, y1, x2, y2 = detection["bbox"]
                roi = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
                if roi.size > 0:
                    detection["dominant_color"] = self._get_dominant_color(roi)
        
        return features, detections
    
    def _get_dominant_color(self, image: np.ndarray) -> List[int]:
        """
        Get the dominant color in an image region.
        
        Args:
            image: Image region as numpy array
            
        Returns:
            List of [B, G, R] values for dominant color
        """
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Calculate average color
        avg_color = np.mean(pixels, axis=0).astype(int).tolist()
        
        return avg_color
    
    def _process_ui(self, ui_state: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process UI state to extract features and detections.
        
        Args:
            ui_state: UI element data
            
        Returns:
            Tuple of (features, detections)
        """
        features = {
            "ui_processed": True,
            "element_count": len(ui_state) if isinstance(ui_state, dict) else 0
        }
        
        detections = []
        
        # Process UI elements
        if isinstance(ui_state, dict):
            for element_id, element_data in ui_state.items():
                element_type = element_data.get("type", "unknown")
                
                detection = {
                    "type": "ui_element",
                    "element_id": element_id,
                    "element_type": element_type,
                    "interactive": element_data.get("interactive", False)
                }
                
                # Extract position if available
                if "position" in element_data:
                    detection["position"] = element_data["position"]
                
                detections.append(detection)
        
        return features, detections
    
    def _process_text(self, text_data: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Process text visible in the input.
        
        Args:
            text_data: Text content to process
            
        Returns:
            Tuple of (features, detections)
        """
        if not isinstance(text_data, str):
            return {"text_processed": False}, []
        
        features = {
            "text_processed": True,
            "text_length": len(text_data),
            "word_count": len(text_data.split())
        }
        
        detections = [
            {
                "type": "text",
                "content": text_data[:100] + "..." if len(text_data) > 100 else text_data,
                "language": self._detect_language(text_data)
            }
        ]
        
        return features, detections
    
    def _detect_language(self, text: str) -> str:
        """
        Simple language detection for text.
        
        Args:
            text: Text to detect language of
            
        Returns:
            Language code (default: 'en')
        """
        # In a real implementation, use a language detection library
        # This is a simple placeholder
        return "en"
    
    def _detect_unusual_patterns(self, detections: List[Dict[str, Any]]) -> bool:
        """
        Detect if there are any unusual patterns in the detections.
        
        Args:
            detections: List of detections from processed input
            
        Returns:
            True if unusual patterns detected, False otherwise
        """
        # Check for objects that rarely occur together
        unusual_combinations = [
            {"dog", "keyboard"},
            {"cat", "sink"},
            {"person", "refrigerator", "bed"}
        ]
        
        detected_classes = {d["class"] for d in detections if "class" in d}
        
        for unusual_combo in unusual_combinations:
            if unusual_combo.issubset(detected_classes):
                return True
        
        # Check for objects in unusual quantities
        class_counts = {}
        for detection in detections:
            if "class" in detection:
                class_counts[detection["class"]] = class_counts.get(detection["class"], 0) + 1
        
        # Unusual to have many of certain objects
        unusual_counts = {
            "person": 10,
            "car": 15,
            "chair": 20
        }
        
        for obj_class, threshold in unusual_counts.items():
            if class_counts.get(obj_class, 0) > threshold:
                return True
        
        return False
    
    def __del__(self):
        """Clean up resources when the agent is destroyed."""
        self._stop_streaming() 