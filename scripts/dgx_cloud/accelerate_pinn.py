#!/usr/bin/env python3
"""
PINN TensorRT Acceleration for DGX Cloud

Converts Physics-Informed Neural Network to TensorRT for <5ms inference.
Target: 20x speedup over CPU for real-time robotics control loops.

Usage:
    python accelerate_pinn.py --export-tensorrt
    python accelerate_pinn.py --benchmark
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent.parent / "models" / "pinn_tensorrt"


def check_tensorrt_availability() -> Dict[str, Any]:
    """Check if TensorRT is available"""
    result = {
        "tensorrt_available": False,
        "torch_available": False,
        "cuda_available": False,
        "version": None
    }
    
    try:
        import torch
        result["torch_available"] = True
        result["cuda_available"] = torch.cuda.is_available()
        
        if result["cuda_available"]:
            logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("⚠️ PyTorch not available")
    
    try:
        import tensorrt as trt
        result["tensorrt_available"] = True
        result["version"] = trt.__version__
        logger.info(f"✅ TensorRT available: v{trt.__version__}")
    except ImportError:
        logger.warning("⚠️ TensorRT not available - will use ONNX export only")
    
    return result


class PINNModel:
    """
    Physics-Informed Neural Network for TensorRT export
    
    Architecture matches UnifiedPhysicsAgent's CMS blocks:
    - 4 levels with different update frequencies
    - Softplus activation (physics-friendly)
    - Multi-domain physics validation
    """
    
    def __init__(self, input_dim: int = 32, hidden_dim: int = 64, num_levels: int = 4):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.model = None
        
    def build_torch_model(self):
        """Build PyTorch model for export"""
        try:
            import torch
            import torch.nn as nn
            
            class PINNNetwork(nn.Module):
                def __init__(self, input_dim, hidden_dim, num_levels):
                    super().__init__()
                    
                    self.input_proj = nn.Linear(input_dim, hidden_dim)
                    
                    # CMS-style blocks for different physics domains
                    self.blocks = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.Softplus(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.Softplus(),
                        )
                        for _ in range(num_levels)
                    ])
                    
                    # Physics output heads
                    self.mechanics_head = nn.Linear(hidden_dim, 6)  # force, torque
                    self.thermo_head = nn.Linear(hidden_dim, 3)     # temp, pressure, entropy
                    self.aero_head = nn.Linear(hidden_dim, 4)       # lift, drag, moment, stall
                    self.valid_head = nn.Linear(hidden_dim, 1)      # overall validity
                    
                def forward(self, x):
                    # Input projection
                    h = self.input_proj(x)
                    
                    # Process through CMS blocks
                    for block in self.blocks:
                        h = h + block(h)  # Residual connection
                    
                    # Physics outputs
                    mechanics = self.mechanics_head(h)
                    thermo = self.thermo_head(h)
                    aero = self.aero_head(h)
                    valid = torch.sigmoid(self.valid_head(h))
                    
                    return {
                        "mechanics": mechanics,
                        "thermodynamics": thermo,
                        "aerodynamics": aero,
                        "validity": valid
                    }
            
            self.model = PINNNetwork(self.input_dim, self.hidden_dim, self.num_levels)
            logger.info(f"✅ Built PINN model: {sum(p.numel() for p in self.model.parameters())} parameters")
            return self.model
            
        except ImportError as e:
            logger.error(f"❌ PyTorch required: {e}")
            return None
    
    def export_onnx(self, output_path: Path) -> bool:
        """Export model to ONNX format"""
        try:
            import torch
            
            if self.model is None:
                self.build_torch_model()
            
            self.model.eval()
            
            # Dummy input
            dummy_input = torch.randn(1, self.input_dim)
            
            # Export
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For dict output, we need to modify the model
            class PINNExportWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    
                def forward(self, x):
                    out = self.model(x)
                    # Concatenate all outputs for ONNX
                    return torch.cat([
                        out["mechanics"],
                        out["thermodynamics"],
                        out["aerodynamics"],
                        out["validity"]
                    ], dim=-1)
            
            wrapper = PINNExportWrapper(self.model)
            
            torch.onnx.export(
                wrapper,
                dummy_input,
                str(output_path),
                input_names=["physics_input"],
                output_names=["physics_output"],
                dynamic_axes={
                    "physics_input": {0: "batch_size"},
                    "physics_output": {0: "batch_size"}
                },
                opset_version=17
            )
            
            logger.info(f"✅ Exported ONNX model to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ ONNX export failed: {e}")
            return False
    
    def export_tensorrt(self, onnx_path: Path, trt_path: Path) -> bool:
        """Convert ONNX to TensorRT engine"""
        try:
            import tensorrt as trt
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for i in range(parser.num_errors):
                        logger.error(f"ONNX parse error: {parser.get_error(i)}")
                    return False
            
            # Build config
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
            
            # Enable FP16 for H100
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("✅ FP16 enabled for H100")
            
            # Build engine
            logger.info("🔧 Building TensorRT engine (this may take a few minutes)...")
            engine = builder.build_serialized_network(network, config)
            
            if engine is None:
                logger.error("❌ TensorRT engine build failed")
                return False
            
            # Save engine
            trt_path.parent.mkdir(parents=True, exist_ok=True)
            with open(trt_path, 'wb') as f:
                f.write(engine)
            
            logger.info(f"✅ TensorRT engine saved to {trt_path}")
            return True
            
        except ImportError:
            logger.warning("⚠️ TensorRT not available - ONNX export only")
            return False
        except Exception as e:
            logger.error(f"❌ TensorRT export failed: {e}")
            return False


def benchmark_pinn(model: PINNModel, iterations: int = 1000) -> Dict[str, Any]:
    """Benchmark PINN inference"""
    try:
        import torch
        
        if model.model is None:
            model.build_torch_model()
        
        model.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.to(device)
        
        # Warmup
        for _ in range(10):
            x = torch.randn(1, model.input_dim, device=device)
            with torch.no_grad():
                _ = model.model(x)
        
        # Benchmark
        latencies = []
        for _ in range(iterations):
            x = torch.randn(1, model.input_dim, device=device)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            with torch.no_grad():
                _ = model.model(x)
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            latencies.append((time.perf_counter() - start) * 1000)
        
        results = {
            "device": str(device),
            "iterations": iterations,
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_per_sec": 1000 / np.mean(latencies)
        }
        
        logger.info(f"📊 PINN Benchmark ({device}):")
        logger.info(f"   Avg latency: {results['avg_latency_ms']:.3f}ms")
        logger.info(f"   P99 latency: {results['p99_latency_ms']:.3f}ms")
        logger.info(f"   Throughput: {results['throughput_per_sec']:.0f}/sec")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="PINN TensorRT Acceleration")
    parser.add_argument("--export-tensorrt", action="store_true", help="Export to TensorRT")
    parser.add_argument("--export-onnx", action="store_true", help="Export to ONNX only")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--iterations", type=int, default=1000, help="Benchmark iterations")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 NIS Protocol - PINN TensorRT Acceleration")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Check availability
    availability = check_tensorrt_availability()
    
    # Create model
    pinn = PINNModel(input_dim=32, hidden_dim=64, num_levels=4)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.export_onnx or args.export_tensorrt:
        # Export ONNX
        onnx_path = OUTPUT_DIR / "pinn_physics.onnx"
        success = pinn.export_onnx(onnx_path)
        
        if success and args.export_tensorrt and availability["tensorrt_available"]:
            # Export TensorRT
            trt_path = OUTPUT_DIR / "pinn_physics.trt"
            pinn.export_tensorrt(onnx_path, trt_path)
    
    if args.benchmark:
        results = benchmark_pinn(pinn, args.iterations)
        
        # Save results
        results_file = OUTPUT_DIR / "benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {results_file}")
    
    if not any([args.export_tensorrt, args.export_onnx, args.benchmark]):
        print("\nUsage:")
        print("  python accelerate_pinn.py --export-onnx      # Export ONNX model")
        print("  python accelerate_pinn.py --export-tensorrt  # Export TensorRT engine")
        print("  python accelerate_pinn.py --benchmark        # Run benchmarks")


if __name__ == "__main__":
    main()
