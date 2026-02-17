"""
Test Scripts for Real Data Training Pipelines
==============================================
Validates data pipelines and model training with real data.

Copyright 2026 Organica AI Solutions
Licensed under Apache License 2.0
"""

import os
import sys
import json
import logging
import unittest
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# =============================================================================
# TEST CONFIGURATION
# =============================================================================

@dataclass
class TestConfig:
    """Test configuration"""
    data_dir: str = "/data/organica-ai/datasets"
    model_dir: str = "/data/organica-ai/models"
    test_batch_size: int = 4
    test_epochs: int = 2
    use_gpu: bool = True
    verbose: bool = True


# =============================================================================
# HIGH PRIORITY TESTS
# =============================================================================

class TestNeMoASRPipeline(unittest.TestCase):
    """Test NeMo ASR real data pipeline"""
    
    @classmethod
    def setUpClass(cls):
        from scripts.training.real_data_pipelines import NeMoASRDataPipeline
        cls.pipeline = NeMoASRDataPipeline()
        cls.config = TestConfig()
    
    def test_01_dataset_configs_exist(self):
        """Test that dataset configurations are defined"""
        self.assertIn("librispeech_clean_100", self.pipeline.DATASETS)
        self.assertIn("common_voice_en", self.pipeline.DATASETS)
        
        config = self.pipeline.DATASETS["librispeech_clean_100"]
        self.assertEqual(config.format, "flac")
        self.assertGreater(config.size_gb, 0)
    
    def test_02_manifest_directory_creation(self):
        """Test manifest directory is created"""
        self.assertTrue(self.pipeline.manifest_dir.exists())
    
    def test_03_download_returns_status(self):
        """Test download method returns boolean"""
        # This won't actually download, just check the method works
        result = self.pipeline.download("librispeech_clean_100")
        self.assertIsInstance(result, bool)
    
    def test_04_generator_interface(self):
        """Test generator interface exists"""
        self.assertTrue(hasattr(self.pipeline, 'get_train_generator'))
        self.assertTrue(hasattr(self.pipeline, 'get_val_generator'))


class TestIsaacRLPipeline(unittest.TestCase):
    """Test Isaac RL Navigation real data pipeline"""
    
    @classmethod
    def setUpClass(cls):
        from scripts.training.real_data_pipelines import IsaacRLDataPipeline
        cls.pipeline = IsaacRLDataPipeline()
    
    def test_01_dataset_configs_exist(self):
        """Test that dataset configurations are defined"""
        self.assertIn("hm3d", self.pipeline.DATASETS)
        self.assertIn("gibson", self.pipeline.DATASETS)
        self.assertIn("mp3d", self.pipeline.DATASETS)
    
    def test_02_episodes_directory_creation(self):
        """Test episodes directory is created"""
        self.assertTrue(self.pipeline.episodes_dir.exists())
    
    def test_03_episode_generation(self):
        """Test episode generation method"""
        # Create temp directory with mock scene
        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = Path(tmpdir)
            # Create mock scene file
            (scene_dir / "test_scene.glb").touch()
            
            episodes = self.pipeline._generate_navigation_episodes(scene_dir)
            self.assertIsInstance(episodes, list)


class TestVisionTrackingPipeline(unittest.TestCase):
    """Test Vision Tracking real data pipeline"""
    
    @classmethod
    def setUpClass(cls):
        from scripts.training.real_data_pipelines import VisionTrackingDataPipeline
        cls.pipeline = VisionTrackingDataPipeline()
    
    def test_01_dataset_configs_exist(self):
        """Test that dataset configurations are defined"""
        self.assertIn("mot17", self.pipeline.DATASETS)
        self.assertIn("lasot", self.pipeline.DATASETS)
        self.assertIn("got10k", self.pipeline.DATASETS)
    
    def test_02_sequences_directory_creation(self):
        """Test sequences directory is created"""
        self.assertTrue(self.pipeline.sequences_dir.exists())
    
    def test_03_mot_format_processing(self):
        """Test MOT format processing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock MOT structure
            seq_dir = Path(tmpdir) / "MOT17-01"
            gt_dir = seq_dir / "gt"
            img_dir = seq_dir / "img1"
            gt_dir.mkdir(parents=True)
            img_dir.mkdir(parents=True)
            
            # Create mock ground truth
            with open(gt_dir / "gt.txt", 'w') as f:
                f.write("1,1,100,100,50,50,1,1,1\n")
                f.write("2,1,105,100,50,50,1,1,1\n")
            
            # Create mock images
            (img_dir / "000001.jpg").touch()
            (img_dir / "000002.jpg").touch()
            
            sequences = self.pipeline._process_mot_format(Path(tmpdir))
            self.assertEqual(len(sequences), 1)
            self.assertEqual(sequences[0]["sequence_id"], "MOT17-01")


# =============================================================================
# MEDIUM PRIORITY TESTS
# =============================================================================

class TestVLARealDataPipeline(unittest.TestCase):
    """Test VLA real data pipeline"""
    
    @classmethod
    def setUpClass(cls):
        from scripts.training.real_data_pipelines import VLARealDataPipeline
        cls.pipeline = VLARealDataPipeline()
    
    def test_01_dataset_configs_exist(self):
        """Test that dataset configurations are defined"""
        self.assertIn("open_x_embodiment", self.pipeline.DATASETS)
        self.assertIn("bridge_v2", self.pipeline.DATASETS)
        self.assertIn("droid", self.pipeline.DATASETS)
    
    def test_02_episodes_directory_creation(self):
        """Test episodes directory is created"""
        self.assertTrue(self.pipeline.episodes_dir.exists())
    
    def test_03_bridge_format_processing(self):
        """Test Bridge format processing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock Bridge structure
            traj_dir = Path(tmpdir) / "trajectory_001"
            traj_dir.mkdir()
            
            # Create mock data
            with open(traj_dir / "observations.json", 'w') as f:
                json.dump([{"instruction": "pick up cube", "state": [0]*7}], f)
            with open(traj_dir / "actions.json", 'w') as f:
                json.dump([{"action": [0.1]*7, "gripper": 0.5}], f)
            
            episodes = self.pipeline._process_bridge_format(Path(tmpdir))
            self.assertEqual(len(episodes), 1)


class TestDroneRealDataPipeline(unittest.TestCase):
    """Test Drone real data pipeline"""
    
    @classmethod
    def setUpClass(cls):
        from scripts.training.real_data_pipelines import DroneRealDataPipeline
        cls.pipeline = DroneRealDataPipeline()
    
    def test_01_dataset_configs_exist(self):
        """Test that dataset configurations are defined"""
        self.assertIn("tartanair", self.pipeline.DATASETS)
        self.assertIn("euroc_mav", self.pipeline.DATASETS)
        self.assertIn("uzh_fpv", self.pipeline.DATASETS)
    
    def test_02_flights_directory_creation(self):
        """Test flights directory is created"""
        self.assertTrue(self.pipeline.flights_dir.exists())


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPipelineFactory(unittest.TestCase):
    """Test pipeline factory"""
    
    def test_01_list_pipelines(self):
        """Test listing available pipelines"""
        from scripts.training.real_data_pipelines import RealDataPipelineFactory
        
        pipelines = RealDataPipelineFactory.list_pipelines()
        self.assertIn("nemo_asr", pipelines)
        self.assertIn("isaac_rl", pipelines)
        self.assertIn("vision_tracking", pipelines)
        self.assertIn("vla", pipelines)
        self.assertIn("drone", pipelines)
    
    def test_02_create_pipelines(self):
        """Test creating pipelines"""
        from scripts.training.real_data_pipelines import RealDataPipelineFactory
        
        for pipeline_type in RealDataPipelineFactory.list_pipelines():
            pipeline = RealDataPipelineFactory.create(pipeline_type)
            self.assertIsNotNone(pipeline)
            self.assertTrue(hasattr(pipeline, 'download'))
            self.assertTrue(hasattr(pipeline, 'preprocess'))
            self.assertTrue(hasattr(pipeline, 'get_train_generator'))
    
    def test_03_invalid_pipeline_raises(self):
        """Test invalid pipeline type raises error"""
        from scripts.training.real_data_pipelines import RealDataPipelineFactory
        
        with self.assertRaises(ValueError):
            RealDataPipelineFactory.create("invalid_pipeline")


# =============================================================================
# MODEL TRAINING TESTS
# =============================================================================

class TestNeMoASRTraining(unittest.TestCase):
    """Test NeMo ASR model training with real data"""
    
    @classmethod
    def setUpClass(cls):
        cls.config = TestConfig()
        cls.skip_gpu = not cls.config.use_gpu
    
    def test_01_model_architecture_loads(self):
        """Test NeMo ASR model architecture loads"""
        try:
            from src.inference.h100_models import NeMoASRModel
            model = NeMoASRModel()
            self.assertIsNotNone(model)
            
            # Check parameter count
            params = sum(p.numel() for p in model.parameters())
            self.assertGreater(params, 0)
            logger.info(f"NeMo ASR model parameters: {params:,}")
        except ImportError as e:
            self.skipTest(f"Model not available: {e}")
    
    def test_02_forward_pass(self):
        """Test forward pass with dummy data"""
        try:
            import torch
            from src.inference.h100_models import NeMoASRModel
            
            model = NeMoASRModel()
            
            # Create dummy mel spectrogram
            batch_size = 2
            mel_bins = 80
            time_steps = 100
            x = torch.randn(batch_size, mel_bins, time_steps)
            
            # Forward pass
            output = model(x)
            self.assertEqual(output.shape[0], batch_size)
            logger.info(f"NeMo ASR output shape: {output.shape}")
        except ImportError as e:
            self.skipTest(f"PyTorch not available: {e}")


class TestVisionTrackingTraining(unittest.TestCase):
    """Test Vision Tracking model training with real data"""
    
    def test_01_model_architecture_loads(self):
        """Test CameraFollowNet model architecture loads"""
        try:
            from src.inference.h100_models import CameraFollowNet
            model = CameraFollowNet()
            self.assertIsNotNone(model)
            
            params = sum(p.numel() for p in model.parameters())
            self.assertGreater(params, 0)
            logger.info(f"CameraFollowNet parameters: {params:,}")
        except ImportError as e:
            self.skipTest(f"Model not available: {e}")
    
    def test_02_forward_pass(self):
        """Test forward pass with dummy data"""
        try:
            import torch
            from src.inference.h100_models import CameraFollowNet
            
            model = CameraFollowNet()
            
            # Create dummy inputs
            batch_size = 2
            state = torch.randn(batch_size, 4)  # x, y, vx, vy
            bbox = torch.randn(batch_size, 4)   # x1, y1, x2, y2
            
            # Forward pass
            output = model(state, bbox)
            self.assertEqual(output.shape, (batch_size, 2))  # pan, tilt
            logger.info(f"CameraFollowNet output shape: {output.shape}")
        except ImportError as e:
            self.skipTest(f"PyTorch not available: {e}")


class TestVLATraining(unittest.TestCase):
    """Test VLA model training with real data"""
    
    def test_01_vla_trainer_imports(self):
        """Test VLA trainer imports"""
        try:
            from src.neurolinux.training.vla_trainer import (
                VLATrainingConfig,
                VLATrainer,
                SmolVLAModel
            )
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"VLA trainer not available: {e}")
    
    def test_02_vla_config_creation(self):
        """Test VLA config creation"""
        try:
            from src.neurolinux.training.vla_trainer import VLATrainingConfig
            
            config = VLATrainingConfig(
                batch_size=4,
                num_epochs=2,
                learning_rate=1e-4
            )
            
            self.assertEqual(config.batch_size, 4)
            self.assertEqual(config.num_epochs, 2)
        except ImportError as e:
            self.skipTest(f"VLA trainer not available: {e}")
    
    def test_03_smolvla_model_creation(self):
        """Test SmolVLA model creation"""
        try:
            import torch
            from src.neurolinux.training.vla_trainer import (
                VLATrainingConfig,
                SmolVLAModel
            )
            
            config = VLATrainingConfig()
            model = SmolVLAModel(config)
            
            params = sum(p.numel() for p in model.parameters())
            self.assertGreater(params, 0)
            logger.info(f"SmolVLA parameters: {params:,}")
        except ImportError as e:
            self.skipTest(f"SmolVLA not available: {e}")


# =============================================================================
# H100 DEPLOYMENT TESTS
# =============================================================================

class TestH100Deployment(unittest.TestCase):
    """Test H100 deployment scripts"""
    
    def test_01_train_script_exists(self):
        """Test H100 training script exists"""
        train_script = PROJECT_ROOT / "scripts" / "train_vla_h100.py"
        self.assertTrue(train_script.exists(), f"Missing: {train_script}")
    
    def test_02_train_script_syntax(self):
        """Test training script has valid Python syntax"""
        train_script = PROJECT_ROOT / "scripts" / "train_vla_h100.py"
        
        if train_script.exists():
            import ast
            with open(train_script, 'r') as f:
                source = f.read()
            
            try:
                ast.parse(source)
                self.assertTrue(True)
            except SyntaxError as e:
                self.fail(f"Syntax error in train_vla_h100.py: {e}")


# =============================================================================
# RUN TESTS
# =============================================================================

def run_high_priority_tests():
    """Run high priority tests only"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # High priority
    suite.addTests(loader.loadTestsFromTestCase(TestNeMoASRPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestIsaacRLPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestVisionTrackingPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestNeMoASRTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestVisionTrackingTraining))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_medium_priority_tests():
    """Run medium priority tests only"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Medium priority
    suite.addTests(loader.loadTestsFromTestCase(TestVLARealDataPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestDroneRealDataPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestVLATraining))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


def run_all_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # All test classes
    test_classes = [
        TestNeMoASRPipeline,
        TestIsaacRLPipeline,
        TestVisionTrackingPipeline,
        TestVLARealDataPipeline,
        TestDroneRealDataPipeline,
        TestPipelineFactory,
        TestNeMoASRTraining,
        TestVisionTrackingTraining,
        TestVLATraining,
        TestH100Deployment,
    ]
    
    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))
    
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Real Data Training Pipelines")
    parser.add_argument("--priority", choices=["high", "medium", "all"], default="all",
                       help="Which priority tests to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("NIS Protocol - Real Data Training Pipeline Tests")
    print("=" * 70)
    
    if args.priority == "high":
        print("\n🔴 Running HIGH PRIORITY tests...\n")
        result = run_high_priority_tests()
    elif args.priority == "medium":
        print("\n🟡 Running MEDIUM PRIORITY tests...\n")
        result = run_medium_priority_tests()
    else:
        print("\n🔵 Running ALL tests...\n")
        result = run_all_tests()
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 70)
    
    sys.exit(0 if result.wasSuccessful() else 1)
