#!/usr/bin/env python3
"""
Prepare Training Data for DGX Cloud BitNet Fine-tuning

Exports telemetry and conversation data from NIS-HUB for H100 training.
Run this before DGX Cloud access to have clean data ready.
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import hashlib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

# Output directories
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "dgx_training"
TELEMETRY_DIR = DATA_DIR / "telemetry"
CONVERSATIONS_DIR = DATA_DIR / "conversations"
ROBOTICS_DIR = DATA_DIR / "robotics"


def ensure_directories():
    """Create output directories"""
    for d in [DATA_DIR, TELEMETRY_DIR, CONVERSATIONS_DIR, ROBOTICS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"📁 Data directories ready: {DATA_DIR}")


def export_bitnet_training_examples() -> Dict[str, Any]:
    """Export existing BitNet training examples"""
    print("\n📊 Exporting BitNet Training Examples...")
    
    results = {"count": 0, "path": None, "status": "not_found"}
    
    training_file = Path("data/bitnet_training/training_examples.json")
    if training_file.exists():
        with open(training_file, 'r') as f:
            examples = json.load(f)
        
        # Convert to training format
        training_data = []
        for ex in examples:
            training_data.append({
                "instruction": ex.get("prompt", ""),
                "output": ex.get("response", ""),
                "quality_score": ex.get("quality_score", 0.7),
                "consciousness_score": ex.get("consciousness_score", 0.7),
                "physics_compliance": ex.get("physics_compliance", 0.7),
            })
        
        output_file = CONVERSATIONS_DIR / "bitnet_examples.jsonl"
        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + "\n")
        
        results["count"] = len(training_data)
        results["path"] = str(output_file)
        results["status"] = "exported"
        print(f"  ✅ Exported {len(training_data)} examples to {output_file}")
    else:
        print(f"  ⚠️ No existing training examples found at {training_file}")
    
    return results


def generate_synthetic_robotics_data(num_samples: int = 1000) -> Dict[str, Any]:
    """Generate synthetic robotics training data"""
    print(f"\n📊 Generating {num_samples} Synthetic Robotics Samples...")
    
    results = {"count": 0, "path": None, "status": "generated"}
    
    training_data = []
    
    # Drone mission scenarios
    mission_types = [
        "survey", "delivery", "inspection", "search_rescue", "mapping"
    ]
    
    for i in range(num_samples):
        mission = np.random.choice(mission_types)
        
        # Generate realistic drone telemetry
        position = np.random.uniform(-1000, 1000, 3).tolist()
        velocity = np.random.uniform(-20, 20, 3).tolist()
        orientation = np.random.uniform(-180, 180, 3).tolist()
        battery = np.random.uniform(20, 100)
        
        # Create instruction-output pair
        instruction = f"Plan a {mission} mission. Current position: {position[:2]}, altitude: {position[2]:.1f}m, battery: {battery:.0f}%"
        
        # Generate appropriate response based on mission type
        if mission == "survey":
            output = f"Initiating survey pattern. Optimal altitude: {abs(position[2]) + 50:.0f}m. Estimated coverage: {np.random.uniform(500, 2000):.0f}m². Battery sufficient for {battery/5:.0f} minutes of operation."
        elif mission == "delivery":
            target = np.random.uniform(-500, 500, 2).tolist()
            output = f"Delivery route calculated. Target: [{target[0]:.1f}, {target[1]:.1f}]. Distance: {np.linalg.norm(np.array(target) - np.array(position[:2])):.0f}m. ETA: {np.random.uniform(1, 10):.1f} minutes."
        elif mission == "inspection":
            output = f"Inspection mode activated. Reducing altitude to {max(10, abs(position[2]) - 30):.0f}m for detailed imaging. Camera resolution: optimal. Stability: engaged."
        elif mission == "search_rescue":
            output = f"Search pattern initiated. Grid size: 100m x 100m. Thermal imaging: active. Estimated search time: {np.random.uniform(10, 60):.0f} minutes."
        else:  # mapping
            output = f"Mapping sequence started. LIDAR: active. Overlap: 70%. Resolution: 5cm/pixel. Estimated completion: {np.random.uniform(15, 45):.0f} minutes."
        
        training_data.append({
            "instruction": instruction,
            "output": output,
            "mission_type": mission,
            "telemetry": {
                "position": position,
                "velocity": velocity,
                "orientation": orientation,
                "battery": battery
            }
        })
    
    output_file = ROBOTICS_DIR / "synthetic_missions.jsonl"
    with open(output_file, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
    
    results["count"] = len(training_data)
    results["path"] = str(output_file)
    print(f"  ✅ Generated {len(training_data)} samples to {output_file}")
    
    return results


def generate_physics_validation_data(num_samples: int = 500) -> Dict[str, Any]:
    """Generate physics validation training data"""
    print(f"\n📊 Generating {num_samples} Physics Validation Samples...")
    
    results = {"count": 0, "path": None, "status": "generated"}
    
    training_data = []
    
    physics_domains = ["mechanics", "thermodynamics", "aerodynamics", "kinematics"]
    
    for i in range(num_samples):
        domain = np.random.choice(physics_domains)
        
        if domain == "mechanics":
            mass = np.random.uniform(0.5, 50)
            force = np.random.uniform(1, 100, 3).tolist()
            instruction = f"Validate force application: mass={mass:.2f}kg, force={force}"
            acceleration = [f/mass for f in force]
            output = f"Physics validated. Acceleration: {[f'{a:.2f}' for a in acceleration]} m/s². Newton's 2nd law satisfied. Max stress: within limits."
            
        elif domain == "thermodynamics":
            temp = np.random.uniform(200, 500)
            pressure = np.random.uniform(50000, 200000)
            instruction = f"Check thermodynamic state: T={temp:.1f}K, P={pressure:.0f}Pa"
            output = f"State validated. Ideal gas assumption: valid. Entropy change: {np.random.uniform(-10, 10):.2f} J/K. Energy conservation: satisfied."
            
        elif domain == "aerodynamics":
            velocity = np.random.uniform(5, 30)
            altitude = np.random.uniform(10, 500)
            instruction = f"Validate flight parameters: velocity={velocity:.1f}m/s, altitude={altitude:.0f}m"
            lift = 0.5 * 1.225 * velocity**2 * 0.5 * 1.5  # Simplified lift
            output = f"Aerodynamics validated. Lift: {lift:.1f}N. Drag coefficient: 0.03. Stall margin: {np.random.uniform(20, 50):.0f}%."
            
        else:  # kinematics
            v0 = np.random.uniform(0, 20, 3).tolist()
            a = np.random.uniform(-5, 5, 3).tolist()
            t = np.random.uniform(1, 10)
            instruction = f"Predict position: v0={v0}, a={a}, t={t:.1f}s"
            pos = [v*t + 0.5*acc*t**2 for v, acc in zip(v0, a)]
            output = f"Kinematics computed. Final position: {[f'{p:.2f}' for p in pos]}m. Final velocity: {[f'{v+acc*t:.2f}' for v, acc in zip(v0, a)]}m/s."
        
        training_data.append({
            "instruction": instruction,
            "output": output,
            "domain": domain,
            "physics_valid": True
        })
    
    output_file = ROBOTICS_DIR / "physics_validation.jsonl"
    with open(output_file, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
    
    results["count"] = len(training_data)
    results["path"] = str(output_file)
    print(f"  ✅ Generated {len(training_data)} samples to {output_file}")
    
    return results


def create_training_manifest() -> Dict[str, Any]:
    """Create manifest of all training data"""
    print("\n📋 Creating Training Manifest...")
    
    manifest = {
        "created": datetime.now().isoformat(),
        "version": "1.0",
        "datasets": [],
        "total_samples": 0,
        "checksum": None
    }
    
    # Scan all JSONL files
    for data_dir in [CONVERSATIONS_DIR, ROBOTICS_DIR, TELEMETRY_DIR]:
        for jsonl_file in data_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r') as f:
                count = sum(1 for _ in f)
            
            # Calculate file checksum
            with open(jsonl_file, 'rb') as f:
                checksum = hashlib.md5(f.read()).hexdigest()
            
            manifest["datasets"].append({
                "path": str(jsonl_file.relative_to(DATA_DIR)),
                "samples": count,
                "checksum": checksum,
                "size_bytes": jsonl_file.stat().st_size
            })
            manifest["total_samples"] += count
    
    # Overall manifest checksum
    manifest["checksum"] = hashlib.md5(
        json.dumps(manifest["datasets"], sort_keys=True).encode()
    ).hexdigest()
    
    manifest_file = DATA_DIR / "manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  ✅ Manifest created: {manifest_file}")
    print(f"  📊 Total samples: {manifest['total_samples']}")
    
    return manifest


def main():
    print("=" * 60)
    print("🚀 NIS Protocol - DGX Cloud Training Data Preparation")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    ensure_directories()
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "exports": {}
    }
    
    # Export existing data
    results["exports"]["bitnet_examples"] = export_bitnet_training_examples()
    
    # Generate synthetic data
    results["exports"]["robotics_missions"] = generate_synthetic_robotics_data(1000)
    results["exports"]["physics_validation"] = generate_physics_validation_data(500)
    
    # Create manifest
    results["manifest"] = create_training_manifest()
    
    # Save preparation report
    report_file = DATA_DIR / "preparation_report.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 DATA PREPARATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal samples prepared: {results['manifest']['total_samples']}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Report saved: {report_file}")
    
    print("\n🎯 Ready for DGX Cloud training!")
    print("   Next step: Run train_bitnet_h100.py on DGX Cloud")
    
    return results


if __name__ == "__main__":
    main()
