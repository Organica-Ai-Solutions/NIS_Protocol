"""
NeuroLinux Edge Deployment Package
Deployment utilities for Pi 5, Jetson, and Drones

Provides:
- Automatic hardware detection
- Model optimization for edge
- Offline mode support
- OTA update mechanism
- Health monitoring

Copyright 2026 Organica AI Solutions
Licensed under Apache License 2.0
"""

import asyncio
import logging
import os
import sys
import time
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np

logger = logging.getLogger("neurolinux.edge_deployment")


class EdgePlatform(Enum):
    """Supported edge platforms"""
    RASPBERRY_PI_5 = "raspberry_pi_5"
    RASPBERRY_PI_4 = "raspberry_pi_4"
    JETSON_NANO = "jetson_nano"
    JETSON_ORIN_NANO = "jetson_orin_nano"
    JETSON_ORIN_NX = "jetson_orin_nx"
    JETSON_AGX_ORIN = "jetson_agx_orin"
    WINDOWS_PC = "windows_pc"
    ALIENWARE_AREA51_R2 = "alienware_area51_r2"
    GENERIC_LINUX = "generic_linux"
    GENERIC_WINDOWS = "generic_windows"
    UNKNOWN = "unknown"


@dataclass
class EdgeCapabilities:
    """Capabilities of edge device"""
    platform: EdgePlatform
    cpu_cores: int = 4
    ram_gb: float = 4.0
    gpu_available: bool = False
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    camera_available: bool = False
    gpio_available: bool = False
    can_available: bool = False
    network_interfaces: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Configuration for edge model deployment"""
    model_name: str
    model_path: str
    quantization: str = "none"  # "none", "int8", "fp16"
    max_batch_size: int = 1
    target_latency_ms: float = 100.0
    memory_limit_mb: float = 512.0


# =============================================================================
# HARDWARE DETECTION
# =============================================================================

class HardwareDetector:
    """Detects edge hardware capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger("neurolinux.hardware_detector")
    
    def detect(self) -> EdgeCapabilities:
        """Detect hardware capabilities"""
        platform = self._detect_platform()
        
        capabilities = EdgeCapabilities(
            platform=platform,
            cpu_cores=self._get_cpu_cores(),
            ram_gb=self._get_ram_gb(),
            gpu_available=self._check_gpu(),
            camera_available=self._check_camera(),
            gpio_available=self._check_gpio(),
            can_available=self._check_can(),
            network_interfaces=self._get_network_interfaces()
        )
        
        # GPU details
        if capabilities.gpu_available:
            gpu_info = self._get_gpu_info()
            capabilities.gpu_name = gpu_info.get("name")
            capabilities.gpu_memory_gb = gpu_info.get("memory_gb")
            capabilities.cuda_available = gpu_info.get("cuda_available", False)
            capabilities.cuda_version = gpu_info.get("cuda_version")
        
        self.logger.info(f"Detected platform: {platform.value}")
        return capabilities
    
    def _detect_platform(self) -> EdgePlatform:
        """Detect edge platform"""
        # Check for Jetson
        try:
            with open("/etc/nv_tegra_release", "r") as f:
                content = f.read()
                if "AGX Orin" in content:
                    return EdgePlatform.JETSON_AGX_ORIN
                elif "Orin NX" in content:
                    return EdgePlatform.JETSON_ORIN_NX
                elif "Orin Nano" in content:
                    return EdgePlatform.JETSON_ORIN_NANO
                else:
                    return EdgePlatform.JETSON_NANO
        except FileNotFoundError:
            pass
        
        # Check for Raspberry Pi
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read()
                if "Raspberry Pi 5" in model:
                    return EdgePlatform.RASPBERRY_PI_5
                elif "Raspberry Pi 4" in model:
                    return EdgePlatform.RASPBERRY_PI_4
        except FileNotFoundError:
            pass
        
        # Check if Linux
        if sys.platform.startswith("linux"):
            return EdgePlatform.GENERIC_LINUX
        
        return EdgePlatform.UNKNOWN
    
    def _get_cpu_cores(self) -> int:
        """Get number of CPU cores"""
        try:
            import multiprocessing
            return multiprocessing.cpu_count()
        except:
            return 4
    
    def _get_ram_gb(self) -> float:
        """Get RAM in GB"""
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / (1024 * 1024)
        except:
            pass
        return 4.0
    
    def _check_gpu(self) -> bool:
        """Check if GPU is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        # Check for Jetson GPU
        try:
            return os.path.exists("/dev/nvhost-gpu")
        except:
            pass
        
        return False
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        info = {}
        
        try:
            import torch
            if torch.cuda.is_available():
                info["name"] = torch.cuda.get_device_name(0)
                info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
                info["cuda_available"] = True
                info["cuda_version"] = torch.version.cuda
        except ImportError:
            pass
        
        return info
    
    def _check_camera(self) -> bool:
        """Check if camera is available"""
        # Check for CSI camera
        if os.path.exists("/dev/video0"):
            return True
        
        # Check for Pi camera
        try:
            import subprocess
            result = subprocess.run(
                ["vcgencmd", "get_camera"],
                capture_output=True,
                text=True
            )
            if "detected=1" in result.stdout:
                return True
        except:
            pass
        
        return False
    
    def _check_gpio(self) -> bool:
        """Check if GPIO is available"""
        return os.path.exists("/sys/class/gpio")
    
    def _check_can(self) -> bool:
        """Check if CAN is available"""
        try:
            import subprocess
            result = subprocess.run(["ip", "link", "show", "can0"], capture_output=True)
            return result.returncode == 0
        except:
            pass
        return False
    
    def _get_network_interfaces(self) -> List[str]:
        """Get network interfaces"""
        interfaces = []
        try:
            import subprocess
            result = subprocess.run(
                ["ip", "-o", "link", "show"],
                capture_output=True,
                text=True
            )
            for line in result.stdout.split("\n"):
                if ":" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        iface = parts[1].strip()
                        if iface not in ["lo"]:
                            interfaces.append(iface)
        except:
            pass
        return interfaces


# =============================================================================
# MODEL OPTIMIZER
# =============================================================================

class EdgeModelOptimizer:
    """Optimizes models for edge deployment"""
    
    def __init__(self, capabilities: EdgeCapabilities):
        self.capabilities = capabilities
        self.logger = logging.getLogger("neurolinux.model_optimizer")
    
    def get_optimal_config(self, model_name: str) -> ModelConfig:
        """Get optimal model configuration for this hardware"""
        
        # VLA models
        if "smolvla" in model_name.lower():
            return self._optimize_smolvla()
        elif "pi0" in model_name.lower():
            return self._optimize_pi0()
        elif "groot" in model_name.lower():
            return self._optimize_groot()
        
        # H100 models
        elif "nemo" in model_name.lower() or "asr" in model_name.lower():
            return self._optimize_asr()
        elif "isaac" in model_name.lower() or "rl" in model_name.lower():
            return self._optimize_rl()
        elif "vision" in model_name.lower() or "tracking" in model_name.lower():
            return self._optimize_vision()
        
        # Default
        return ModelConfig(
            model_name=model_name,
            model_path=f"models/{model_name}",
            quantization="fp16" if self.capabilities.gpu_available else "int8"
        )
    
    def _optimize_smolvla(self) -> ModelConfig:
        """Optimize SmolVLA for edge"""
        if self.capabilities.platform in [EdgePlatform.RASPBERRY_PI_5, EdgePlatform.RASPBERRY_PI_4]:
            return ModelConfig(
                model_name="smolvla",
                model_path="models/smolvla_int8.onnx",
                quantization="int8",
                max_batch_size=1,
                target_latency_ms=200.0,
                memory_limit_mb=512.0
            )
        elif self.capabilities.platform in [EdgePlatform.JETSON_NANO]:
            return ModelConfig(
                model_name="smolvla",
                model_path="models/smolvla_fp16.engine",
                quantization="fp16",
                max_batch_size=1,
                target_latency_ms=100.0,
                memory_limit_mb=1024.0
            )
        else:  # Jetson Orin
            return ModelConfig(
                model_name="smolvla",
                model_path="models/smolvla_fp16.engine",
                quantization="fp16",
                max_batch_size=4,
                target_latency_ms=50.0,
                memory_limit_mb=2048.0
            )
    
    def _optimize_pi0(self) -> ModelConfig:
        """Optimize PI0 for edge"""
        if self.capabilities.platform.value.startswith("jetson_orin"):
            return ModelConfig(
                model_name="pi0",
                model_path="models/pi0_fp16.engine",
                quantization="fp16",
                max_batch_size=1,
                target_latency_ms=100.0,
                memory_limit_mb=4096.0
            )
        else:
            # PI0 too large for Pi/Nano, fallback to SmolVLA
            self.logger.warning("PI0 too large for this platform, using SmolVLA")
            return self._optimize_smolvla()
    
    def _optimize_groot(self) -> ModelConfig:
        """Optimize GR00T for edge"""
        if self.capabilities.platform == EdgePlatform.JETSON_AGX_ORIN:
            return ModelConfig(
                model_name="groot",
                model_path="models/groot_fp16.engine",
                quantization="fp16",
                max_batch_size=1,
                target_latency_ms=150.0,
                memory_limit_mb=8192.0
            )
        else:
            # GR00T needs AGX Orin, fallback
            self.logger.warning("GR00T requires AGX Orin, using SmolVLA")
            return self._optimize_smolvla()
    
    def _optimize_asr(self) -> ModelConfig:
        """Optimize ASR for edge"""
        return ModelConfig(
            model_name="nemo_asr",
            model_path="models/nemo_asr_int8.onnx",
            quantization="int8",
            max_batch_size=1,
            target_latency_ms=500.0,
            memory_limit_mb=256.0
        )
    
    def _optimize_rl(self) -> ModelConfig:
        """Optimize RL policy for edge"""
        return ModelConfig(
            model_name="isaac_rl",
            model_path="models/isaac_rl_fp32.onnx",
            quantization="none",  # RL policies are small
            max_batch_size=1,
            target_latency_ms=1.0,  # Real-time
            memory_limit_mb=64.0
        )
    
    def _optimize_vision(self) -> ModelConfig:
        """Optimize vision tracking for edge"""
        if self.capabilities.gpu_available:
            return ModelConfig(
                model_name="vision_tracking",
                model_path="models/vision_tracking_fp16.engine",
                quantization="fp16",
                max_batch_size=1,
                target_latency_ms=33.0,  # 30 FPS
                memory_limit_mb=512.0
            )
        else:
            return ModelConfig(
                model_name="vision_tracking",
                model_path="models/vision_tracking_int8.onnx",
                quantization="int8",
                max_batch_size=1,
                target_latency_ms=100.0,
                memory_limit_mb=256.0
            )
    
    async def convert_model(
        self,
        source_path: str,
        config: ModelConfig,
        output_dir: str = "models"
    ) -> str:
        """Convert model to optimized format"""
        os.makedirs(output_dir, exist_ok=True)
        
        if config.quantization == "int8":
            return await self._convert_to_int8(source_path, output_dir)
        elif config.quantization == "fp16":
            return await self._convert_to_fp16(source_path, output_dir)
        else:
            return source_path
    
    async def _convert_to_int8(self, source_path: str, output_dir: str) -> str:
        """Convert to INT8 ONNX"""
        try:
            import onnx
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            output_path = os.path.join(output_dir, f"{Path(source_path).stem}_int8.onnx")
            
            quantize_dynamic(
                source_path,
                output_path,
                weight_type=QuantType.QInt8
            )
            
            self.logger.info(f"Converted to INT8: {output_path}")
            return output_path
            
        except ImportError:
            self.logger.warning("ONNX quantization not available")
            return source_path
    
    async def _convert_to_fp16(self, source_path: str, output_dir: str) -> str:
        """Convert to FP16 TensorRT engine"""
        if not self.capabilities.cuda_available:
            self.logger.warning("CUDA not available for TensorRT conversion")
            return source_path
        
        try:
            import tensorrt as trt
            
            output_path = os.path.join(output_dir, f"{Path(source_path).stem}_fp16.engine")
            
            # TensorRT conversion would go here
            # This is a placeholder - actual conversion requires more setup
            
            self.logger.info(f"Would convert to FP16 TensorRT: {output_path}")
            return source_path  # Return source for now
            
        except ImportError:
            self.logger.warning("TensorRT not available")
            return source_path


# =============================================================================
# OFFLINE MODE MANAGER
# =============================================================================

class OfflineModeManager:
    """Manages offline operation when cloud is unavailable"""
    
    def __init__(self, cache_dir: str = "/var/cache/neurolinux"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger("neurolinux.offline")
        
        self._offline_mode = False
        self._cached_plans: Dict[str, Dict] = {}
        self._last_sync = 0.0
    
    def is_offline(self) -> bool:
        """Check if in offline mode"""
        return self._offline_mode
    
    async def check_connectivity(self, url: str = "http://localhost:8000/health") -> bool:
        """Check cloud connectivity"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        self._offline_mode = False
                        return True
        except:
            pass
        
        self._offline_mode = True
        return False
    
    def cache_plan(self, goal: str, plan: Dict[str, Any]):
        """Cache plan for offline use"""
        key = self._hash_goal(goal)
        self._cached_plans[key] = plan
        
        # Save to disk
        cache_file = self.cache_dir / f"plan_{key}.json"
        with open(cache_file, "w") as f:
            json.dump({"goal": goal, "plan": plan}, f)
    
    def get_cached_plan(self, goal: str) -> Optional[Dict[str, Any]]:
        """Get cached plan for goal"""
        key = self._hash_goal(goal)
        
        # Check memory cache
        if key in self._cached_plans:
            return self._cached_plans[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"plan_{key}.json"
        if cache_file.exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
                self._cached_plans[key] = data["plan"]
                return data["plan"]
        
        return None
    
    def _hash_goal(self, goal: str) -> str:
        """Hash goal for cache key"""
        return hashlib.md5(goal.lower().encode()).hexdigest()[:16]
    
    def cache_model(self, model_name: str, model_path: str):
        """Cache model for offline use"""
        cache_path = self.cache_dir / "models" / model_name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy model to cache
        import shutil
        if os.path.exists(model_path):
            shutil.copy2(model_path, cache_path)
            self.logger.info(f"Cached model: {model_name}")
    
    def get_cached_model(self, model_name: str) -> Optional[str]:
        """Get cached model path"""
        cache_path = self.cache_dir / "models" / model_name
        if cache_path.exists():
            return str(cache_path)
        return None


# =============================================================================
# HEALTH MONITOR
# =============================================================================

class EdgeHealthMonitor:
    """Monitors edge device health"""
    
    def __init__(self):
        self.logger = logging.getLogger("neurolinux.health")
        
        self._metrics: Dict[str, List[float]] = {
            "cpu_percent": [],
            "memory_percent": [],
            "gpu_percent": [],
            "temperature_c": [],
            "inference_latency_ms": []
        }
        self._max_history = 100
    
    async def collect_metrics(self) -> Dict[str, float]:
        """Collect current health metrics"""
        metrics = {}
        
        # CPU usage
        try:
            import psutil
            metrics["cpu_percent"] = psutil.cpu_percent()
            metrics["memory_percent"] = psutil.virtual_memory().percent
        except ImportError:
            metrics["cpu_percent"] = 0.0
            metrics["memory_percent"] = 0.0
        
        # GPU usage (Jetson)
        try:
            with open("/sys/devices/gpu.0/load", "r") as f:
                metrics["gpu_percent"] = float(f.read().strip()) / 10.0
        except:
            metrics["gpu_percent"] = 0.0
        
        # Temperature
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                metrics["temperature_c"] = float(f.read().strip()) / 1000.0
        except:
            metrics["temperature_c"] = 0.0
        
        # Update history
        for key, value in metrics.items():
            if key in self._metrics:
                self._metrics[key].append(value)
                if len(self._metrics[key]) > self._max_history:
                    self._metrics[key] = self._metrics[key][-self._max_history:]
        
        return metrics
    
    def record_inference_latency(self, latency_ms: float):
        """Record inference latency"""
        self._metrics["inference_latency_ms"].append(latency_ms)
        if len(self._metrics["inference_latency_ms"]) > self._max_history:
            self._metrics["inference_latency_ms"] = self._metrics["inference_latency_ms"][-self._max_history:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        metrics = {
            key: np.mean(values) if values else 0.0
            for key, values in self._metrics.items()
        }
        
        # Determine health status
        status = "healthy"
        warnings = []
        
        if metrics.get("cpu_percent", 0) > 90:
            status = "warning"
            warnings.append("High CPU usage")
        
        if metrics.get("memory_percent", 0) > 90:
            status = "warning"
            warnings.append("High memory usage")
        
        if metrics.get("temperature_c", 0) > 80:
            status = "critical"
            warnings.append("High temperature")
        
        return {
            "status": status,
            "metrics": metrics,
            "warnings": warnings,
            "timestamp": time.time()
        }


# =============================================================================
# EDGE DEPLOYMENT MANAGER
# =============================================================================

class EdgeDeploymentManager:
    """
    Main manager for edge deployment.
    Coordinates hardware detection, model optimization, and health monitoring.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger("neurolinux.deployment")
        
        # Components
        self.hardware_detector = HardwareDetector()
        self.capabilities: Optional[EdgeCapabilities] = None
        self.model_optimizer: Optional[EdgeModelOptimizer] = None
        self.offline_manager = OfflineModeManager()
        self.health_monitor = EdgeHealthMonitor()
        
        # Configuration
        self.config_path = config_path or "/etc/neurolinux/config.json"
        self.config: Dict[str, Any] = {}
        
        # State
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize edge deployment"""
        try:
            # Detect hardware
            self.capabilities = self.hardware_detector.detect()
            self.model_optimizer = EdgeModelOptimizer(self.capabilities)
            
            # Load configuration
            self._load_config()
            
            # Check connectivity
            await self.offline_manager.check_connectivity()
            
            self._initialized = True
            self.logger.info(f"Edge deployment initialized: {self.capabilities.platform.value}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def _load_config(self):
        """Load configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as f:
                    self.config = json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load config: {e}")
    
    def save_config(self):
        """Save configuration"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save config: {e}")
    
    def get_optimal_model_config(self, model_name: str) -> ModelConfig:
        """Get optimal model configuration"""
        if not self.model_optimizer:
            raise RuntimeError("Not initialized")
        return self.model_optimizer.get_optimal_config(model_name)
    
    async def prepare_model(self, model_name: str, source_path: str) -> str:
        """Prepare model for edge deployment"""
        if not self.model_optimizer:
            raise RuntimeError("Not initialized")
        
        config = self.get_optimal_model_config(model_name)
        optimized_path = await self.model_optimizer.convert_model(source_path, config)
        
        # Cache for offline use
        self.offline_manager.cache_model(model_name, optimized_path)
        
        return optimized_path
    
    async def get_health(self) -> Dict[str, Any]:
        """Get device health status"""
        metrics = await self.health_monitor.collect_metrics()
        status = self.health_monitor.get_health_status()
        
        return {
            **status,
            "platform": self.capabilities.platform.value if self.capabilities else "unknown",
            "offline_mode": self.offline_manager.is_offline()
        }
    
    def get_deployment_info(self) -> Dict[str, Any]:
        """Get deployment information"""
        return {
            "initialized": self._initialized,
            "platform": self.capabilities.platform.value if self.capabilities else "unknown",
            "capabilities": {
                "cpu_cores": self.capabilities.cpu_cores if self.capabilities else 0,
                "ram_gb": self.capabilities.ram_gb if self.capabilities else 0,
                "gpu_available": self.capabilities.gpu_available if self.capabilities else False,
                "gpu_name": self.capabilities.gpu_name if self.capabilities else None,
                "cuda_available": self.capabilities.cuda_available if self.capabilities else False
            },
            "offline_mode": self.offline_manager.is_offline(),
            "config_path": self.config_path
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_edge_deployment_manager(config_path: Optional[str] = None) -> EdgeDeploymentManager:
    """Create edge deployment manager"""
    return EdgeDeploymentManager(config_path)


async def auto_setup_edge() -> EdgeDeploymentManager:
    """Auto-setup edge deployment"""
    manager = create_edge_deployment_manager()
    await manager.initialize()
    return manager
