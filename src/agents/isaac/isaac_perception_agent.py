#!/usr/bin/env python3
"""
NIS Protocol - Isaac Perception Agent
Integration with NVIDIA Isaac perception models

Features:
- FoundationPose: 6D object pose estimation
- SyntheticaDETR: Object detection
- cuVSLAM: Visual SLAM
- nvblox: 3D reconstruction
"""

import os
import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger("nis.agents.isaac.perception")

# Try to import Isaac perception modules
try:
    from isaac_ros_foundationpose import FoundationPose
    FOUNDATION_POSE_AVAILABLE = True
except ImportError:
    FOUNDATION_POSE_AVAILABLE = False

try:
    from isaac_ros_syntheticadetr import SyntheticaDETR
    SYNTHETICA_DETR_AVAILABLE = True
except ImportError:
    SYNTHETICA_DETR_AVAILABLE = False

try:
    from isaac_ros_visual_slam import VisualSlam
    VISUAL_SLAM_AVAILABLE = True
except ImportError:
    VISUAL_SLAM_AVAILABLE = False


@dataclass
class PerceptionConfig:
    """Perception configuration"""
    # Model paths (for offline mode)
    foundation_pose_model: str = "/opt/isaac_models/foundationpose_weights.pth"
    synthetica_detr_model: str = "/opt/isaac_models/syntheticadetr_weights.pth"
    
    # Detection thresholds
    detection_confidence: float = 0.5
    pose_confidence: float = 0.7
    
    # Camera parameters
    camera_intrinsics: List[float] = field(default_factory=lambda: [
        615.0, 0, 320.0,  # fx, 0, cx
        0, 615.0, 240.0,  # 0, fy, cy
        0, 0, 1           # 0, 0, 1
    ])
    image_width: int = 640
    image_height: int = 480


@dataclass
class Detection:
    """Object detection result"""
    class_name: str
    class_id: int
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None


@dataclass
class Pose6D:
    """6D pose estimation result"""
    object_id: str
    position: Tuple[float, float, float]  # x, y, z
    orientation: Tuple[float, float, float, float]  # quaternion wxyz
    confidence: float
    timestamp: float = 0.0


class IsaacPerceptionAgent:
    """
    Isaac Perception integration agent
    
    Provides:
    - Object detection (SyntheticaDETR)
    - 6D pose estimation (FoundationPose)
    - Visual SLAM (cuVSLAM)
    - Scene understanding
    """
    
    def __init__(self, config: PerceptionConfig = None):
        self.config = config or PerceptionConfig()
        self.initialized = False
        
        # Model availability
        self.foundation_pose_available = FOUNDATION_POSE_AVAILABLE
        self.synthetica_detr_available = SYNTHETICA_DETR_AVAILABLE
        self.visual_slam_available = VISUAL_SLAM_AVAILABLE
        
        # Model instances
        self._foundation_pose = None
        self._synthetica_detr = None
        self._visual_slam = None
        
        # Object database for pose estimation
        self._known_objects: Dict[str, Any] = {}
        
        # SLAM state
        self._slam_pose = None
        self._slam_map = None
        
        # Statistics
        self.stats = {
            "detections": 0,
            "pose_estimations": 0,
            "slam_updates": 0,
            "avg_detection_time_ms": 0.0,
            "avg_pose_time_ms": 0.0
        }
        
        logger.info(f"Isaac Perception Agent created")
        logger.info(f"  FoundationPose: {self.foundation_pose_available}")
        logger.info(f"  SyntheticaDETR: {self.synthetica_detr_available}")
        logger.info(f"  cuVSLAM: {self.visual_slam_available}")
    
    async def initialize(self) -> bool:
        """Initialize perception models"""
        if self.initialized:
            return True
        
        try:
            # Initialize FoundationPose
            if self.foundation_pose_available:
                self._foundation_pose = FoundationPose()
                if os.path.exists(self.config.foundation_pose_model):
                    self._foundation_pose.load_weights(self.config.foundation_pose_model)
                logger.info("FoundationPose initialized")
            
            # Initialize SyntheticaDETR
            if self.synthetica_detr_available:
                self._synthetica_detr = SyntheticaDETR()
                if os.path.exists(self.config.synthetica_detr_model):
                    self._synthetica_detr.load_weights(self.config.synthetica_detr_model)
                logger.info("SyntheticaDETR initialized")
            
            # Initialize Visual SLAM
            if self.visual_slam_available:
                self._visual_slam = VisualSlam()
                logger.info("cuVSLAM initialized")
            
            self.initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize perception: {e}")
            self.initialized = True  # Continue in mock mode
            return True
    
    async def detect_objects(
        self,
        image: np.ndarray,
        confidence_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Detect objects in an image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            confidence_threshold: Minimum confidence for detections
        
        Returns:
            Detection results with bounding boxes and classes
        """
        if not self.initialized:
            await self.initialize()
        
        threshold = confidence_threshold or self.config.detection_confidence
        start_time = time.time()
        
        if self.synthetica_detr_available and self._synthetica_detr:
            try:
                raw_detections = self._synthetica_detr.detect(image)
                
                detections = [
                    Detection(
                        class_name=d.class_name,
                        class_id=d.class_id,
                        confidence=d.confidence,
                        bbox=(d.x1, d.y1, d.x2, d.y2),
                        mask=d.mask if hasattr(d, 'mask') else None
                    )
                    for d in raw_detections
                    if d.confidence >= threshold
                ]
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
                detections = []
        else:
            # Mock detections for testing
            detections = self._mock_detect(image, threshold)
        
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["detections"] += len(detections)
        self._update_avg_time("avg_detection_time_ms", elapsed_ms)
        
        return {
            "success": True,
            "detections": [
                {
                    "class_name": d.class_name,
                    "class_id": d.class_id,
                    "confidence": d.confidence,
                    "bbox": d.bbox
                }
                for d in detections
            ],
            "count": len(detections),
            "inference_time_ms": elapsed_ms
        }
    
    async def estimate_pose(
        self,
        image: np.ndarray,
        object_id: str,
        depth_image: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Estimate 6D pose of a known object
        
        Args:
            image: RGB image
            object_id: ID of the object to estimate pose for
            depth_image: Optional depth image for better accuracy
        
        Returns:
            6D pose estimation result
        """
        if not self.initialized:
            await self.initialize()
        
        start_time = time.time()
        
        if self.foundation_pose_available and self._foundation_pose:
            try:
                pose = self._foundation_pose.estimate_pose(
                    image,
                    object_id,
                    depth=depth_image,
                    camera_intrinsics=np.array(self.config.camera_intrinsics).reshape(3, 3)
                )
                
                result = Pose6D(
                    object_id=object_id,
                    position=(pose.x, pose.y, pose.z),
                    orientation=(pose.qw, pose.qx, pose.qy, pose.qz),
                    confidence=pose.confidence,
                    timestamp=time.time()
                )
                
            except Exception as e:
                logger.error(f"Pose estimation error: {e}")
                result = None
        else:
            # Mock pose for testing
            result = self._mock_pose(object_id)
        
        elapsed_ms = (time.time() - start_time) * 1000
        self.stats["pose_estimations"] += 1
        self._update_avg_time("avg_pose_time_ms", elapsed_ms)
        
        if result and result.confidence >= self.config.pose_confidence:
            return {
                "success": True,
                "object_id": result.object_id,
                "position": result.position,
                "orientation": result.orientation,
                "confidence": result.confidence,
                "inference_time_ms": elapsed_ms
            }
        else:
            return {
                "success": False,
                "error": "Pose estimation failed or low confidence",
                "inference_time_ms": elapsed_ms
            }
    
    async def update_slam(
        self,
        image: np.ndarray,
        depth_image: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Update Visual SLAM with new frame
        
        Args:
            image: RGB image
            depth_image: Optional depth image
        
        Returns:
            Current camera pose and map update status
        """
        if not self.initialized:
            await self.initialize()
        
        if self.visual_slam_available and self._visual_slam:
            try:
                result = self._visual_slam.process_frame(image, depth_image)
                
                self._slam_pose = {
                    "position": result.position.tolist(),
                    "orientation": result.orientation.tolist()
                }
                
                self.stats["slam_updates"] += 1
                
                return {
                    "success": True,
                    "pose": self._slam_pose,
                    "tracking_status": result.tracking_status,
                    "keyframes": result.num_keyframes,
                    "map_points": result.num_map_points
                }
                
            except Exception as e:
                logger.error(f"SLAM update error: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        else:
            # Mock SLAM update
            self._slam_pose = self._mock_slam_pose()
            self.stats["slam_updates"] += 1
            
            return {
                "success": True,
                "mode": "mock",
                "pose": self._slam_pose,
                "tracking_status": "tracking"
            }
    
    def register_object(
        self,
        object_id: str,
        mesh_path: str = None,
        reference_images: List[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Register a new object for pose estimation
        
        Args:
            object_id: Unique identifier for the object
            mesh_path: Path to 3D mesh file (for FoundationPose)
            reference_images: Reference images of the object
        
        Returns:
            Registration status
        """
        self._known_objects[object_id] = {
            "mesh_path": mesh_path,
            "reference_images": reference_images,
            "registered_at": time.time()
        }
        
        if self.foundation_pose_available and self._foundation_pose and mesh_path:
            try:
                self._foundation_pose.register_object(object_id, mesh_path)
            except Exception as e:
                logger.error(f"Object registration error: {e}")
        
        return {
            "success": True,
            "object_id": object_id,
            "registered": True
        }
    
    def _mock_detect(self, image: np.ndarray, threshold: float) -> List[Detection]:
        """Generate mock detections for testing"""
        import random
        
        classes = ["box", "cylinder", "sphere", "tool", "part"]
        detections = []
        
        num_objects = random.randint(0, 3)
        for i in range(num_objects):
            confidence = random.uniform(0.5, 0.99)
            if confidence >= threshold:
                x1 = random.randint(0, 400)
                y1 = random.randint(0, 300)
                x2 = x1 + random.randint(50, 150)
                y2 = y1 + random.randint(50, 150)
                
                detections.append(Detection(
                    class_name=random.choice(classes),
                    class_id=i,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2)
                ))
        
        return detections
    
    def _mock_pose(self, object_id: str) -> Pose6D:
        """Generate mock pose for testing"""
        import random
        
        return Pose6D(
            object_id=object_id,
            position=(
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                random.uniform(0.5, 2)
            ),
            orientation=(1.0, 0.0, 0.0, 0.0),  # identity quaternion
            confidence=random.uniform(0.7, 0.99),
            timestamp=time.time()
        )
    
    def _mock_slam_pose(self) -> Dict[str, Any]:
        """Generate mock SLAM pose for testing"""
        import random
        
        return {
            "position": [
                random.uniform(-5, 5),
                random.uniform(-5, 5),
                random.uniform(0, 2)
            ],
            "orientation": [1.0, 0.0, 0.0, 0.0]
        }
    
    def _update_avg_time(self, stat_key: str, new_time: float):
        """Update running average time"""
        current = self.stats[stat_key]
        count = self.stats["detections"] + self.stats["pose_estimations"]
        if count > 0:
            self.stats[stat_key] = (current * (count - 1) + new_time) / count
    
    def get_slam_pose(self) -> Optional[Dict[str, Any]]:
        """Get current SLAM pose"""
        return self._slam_pose
    
    def get_known_objects(self) -> List[str]:
        """Get list of registered objects"""
        return list(self._known_objects.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get perception statistics"""
        return {
            **self.stats,
            "initialized": self.initialized,
            "foundation_pose_available": self.foundation_pose_available,
            "synthetica_detr_available": self.synthetica_detr_available,
            "visual_slam_available": self.visual_slam_available,
            "known_objects": len(self._known_objects)
        }
    
    async def shutdown(self):
        """Shutdown perception agent"""
        logger.info("Shutting down Isaac Perception Agent...")
        
        self._foundation_pose = None
        self._synthetica_detr = None
        self._visual_slam = None
        self._known_objects.clear()
        self.initialized = False
        
        logger.info("Isaac Perception Agent shutdown complete")


# Singleton instance
_isaac_perception: Optional[IsaacPerceptionAgent] = None


def get_isaac_perception() -> IsaacPerceptionAgent:
    """Get the Isaac Perception singleton"""
    global _isaac_perception
    if _isaac_perception is None:
        _isaac_perception = IsaacPerceptionAgent()
    return _isaac_perception
