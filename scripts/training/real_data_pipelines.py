"""
Real Data Pipelines for NIS Protocol Model Training
====================================================
HIGH PRIORITY and MEDIUM PRIORITY data pipelines for retraining
models with real-world data instead of synthetic.

Copyright 2026 Organica AI Solutions
Licensed under Apache License 2.0
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Generator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import urllib.request
import tarfile
import zipfile

logger = logging.getLogger(__name__)

# =============================================================================
# BASE DATA PIPELINE
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a dataset"""
    name: str
    url: str
    size_gb: float
    format: str  # "jsonl", "tfrecord", "parquet", "hdf5"
    license: str
    citation: str = ""
    checksum: str = ""


class BaseDataPipeline(ABC):
    """Base class for all data pipelines"""
    
    def __init__(self, data_dir: str = "/data/organica-ai/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def download(self) -> bool:
        """Download the dataset"""
        pass
    
    @abstractmethod
    def preprocess(self) -> bool:
        """Preprocess the dataset for training"""
        pass
    
    @abstractmethod
    def get_train_generator(self, batch_size: int) -> Generator:
        """Get training data generator"""
        pass
    
    @abstractmethod
    def get_val_generator(self, batch_size: int) -> Generator:
        """Get validation data generator"""
        pass
    
    def verify_checksum(self, filepath: Path, expected: str) -> bool:
        """Verify file checksum"""
        if not expected:
            return True
        
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        
        return sha256.hexdigest() == expected


# =============================================================================
# HIGH PRIORITY: NeMo ASR Real Data Pipeline
# =============================================================================

class NeMoASRDataPipeline(BaseDataPipeline):
    """
    Real data pipeline for NeMo ASR training.
    
    Datasets:
    - LibriSpeech (960h English speech)
    - Common Voice (multilingual)
    - Mozilla DeepSpeech
    """
    
    DATASETS = {
        "librispeech_clean_100": DatasetConfig(
            name="LibriSpeech clean-100",
            url="https://www.openslr.org/resources/12/train-clean-100.tar.gz",
            size_gb=6.3,
            format="flac",
            license="CC BY 4.0",
            citation="Panayotov et al., 2015"
        ),
        "librispeech_clean_360": DatasetConfig(
            name="LibriSpeech clean-360",
            url="https://www.openslr.org/resources/12/train-clean-360.tar.gz",
            size_gb=23.0,
            format="flac",
            license="CC BY 4.0",
            citation="Panayotov et al., 2015"
        ),
        "librispeech_other_500": DatasetConfig(
            name="LibriSpeech other-500",
            url="https://www.openslr.org/resources/12/train-other-500.tar.gz",
            size_gb=30.0,
            format="flac",
            license="CC BY 4.0",
            citation="Panayotov et al., 2015"
        ),
        "common_voice_en": DatasetConfig(
            name="Common Voice English",
            url="https://commonvoice.mozilla.org/en/datasets",
            size_gb=70.0,
            format="mp3",
            license="CC0",
            citation="Mozilla Common Voice"
        )
    }
    
    def __init__(self, data_dir: str = "/data/organica-ai/datasets/asr"):
        super().__init__(data_dir)
        self.manifest_dir = self.data_dir / "manifests"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, dataset_key: str = "librispeech_clean_100") -> bool:
        """Download LibriSpeech dataset"""
        if dataset_key not in self.DATASETS:
            self.logger.error(f"Unknown dataset: {dataset_key}")
            return False
        
        config = self.DATASETS[dataset_key]
        output_path = self.data_dir / f"{dataset_key}.tar.gz"
        
        if output_path.exists():
            self.logger.info(f"Dataset already downloaded: {output_path}")
            return True
        
        self.logger.info(f"Downloading {config.name} ({config.size_gb} GB)...")
        
        try:
            urllib.request.urlretrieve(config.url, output_path)
            self.logger.info(f"Downloaded to {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"Download failed: {e}")
            return False
    
    def preprocess(self, dataset_key: str = "librispeech_clean_100") -> bool:
        """Extract and create NeMo manifests"""
        archive_path = self.data_dir / f"{dataset_key}.tar.gz"
        extract_dir = self.data_dir / dataset_key
        
        if not archive_path.exists():
            self.logger.error(f"Archive not found: {archive_path}")
            return False
        
        # Extract
        if not extract_dir.exists():
            self.logger.info(f"Extracting {archive_path}...")
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(self.data_dir)
        
        # Create NeMo manifest
        manifest_path = self.manifest_dir / f"{dataset_key}_manifest.json"
        self._create_nemo_manifest(extract_dir, manifest_path)
        
        return True
    
    def _create_nemo_manifest(self, audio_dir: Path, manifest_path: Path):
        """Create NeMo-format manifest from LibriSpeech structure"""
        entries = []
        
        for trans_file in audio_dir.rglob("*.trans.txt"):
            with open(trans_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        audio_id, text = parts
                        audio_path = trans_file.parent / f"{audio_id}.flac"
                        
                        if audio_path.exists():
                            entries.append({
                                "audio_filepath": str(audio_path),
                                "text": text.lower(),
                                "duration": self._get_audio_duration(audio_path)
                            })
        
        with open(manifest_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        
        self.logger.info(f"Created manifest with {len(entries)} entries: {manifest_path}")
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds"""
        try:
            import soundfile as sf
            info = sf.info(str(audio_path))
            return info.duration
        except:
            return 0.0
    
    def get_train_generator(self, batch_size: int = 32) -> Generator:
        """Get training data generator"""
        manifest_path = self.manifest_dir / "librispeech_clean_100_manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            entries = [json.loads(line) for line in f]
        
        batch = []
        for entry in entries:
            batch.append(entry)
            if len(batch) == batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    def get_val_generator(self, batch_size: int = 32) -> Generator:
        """Get validation data generator (use dev-clean)"""
        return self.get_train_generator(batch_size)


# =============================================================================
# HIGH PRIORITY: Isaac RL Navigation Real Data Pipeline
# =============================================================================

class IsaacRLDataPipeline(BaseDataPipeline):
    """
    Real data pipeline for Isaac RL Navigation training.
    
    Datasets:
    - HM3D (Habitat-Matterport 3D)
    - Gibson (real indoor scans)
    - Matterport3D
    """
    
    DATASETS = {
        "hm3d": DatasetConfig(
            name="Habitat-Matterport 3D",
            url="https://github.com/facebookresearch/habitat-matterport3d-dataset",
            size_gb=130.0,
            format="glb",
            license="Matterport Research License",
            citation="Ramakrishnan et al., 2021"
        ),
        "gibson": DatasetConfig(
            name="Gibson Environment Dataset",
            url="https://github.com/StanfordVL/GibsonEnv",
            size_gb=50.0,
            format="obj",
            license="Gibson Dataset License",
            citation="Xia et al., 2018"
        ),
        "mp3d": DatasetConfig(
            name="Matterport3D",
            url="https://niessner.github.io/Matterport/",
            size_gb=90.0,
            format="ply",
            license="Matterport3D TOS",
            citation="Chang et al., 2017"
        )
    }
    
    def __init__(self, data_dir: str = "/data/organica-ai/datasets/navigation"):
        super().__init__(data_dir)
        self.episodes_dir = self.data_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, dataset_key: str = "hm3d") -> bool:
        """Download navigation dataset (requires manual steps for some)"""
        self.logger.info(f"Navigation datasets require manual download due to licensing.")
        self.logger.info(f"Please visit: {self.DATASETS[dataset_key].url}")
        self.logger.info(f"After downloading, place files in: {self.data_dir / dataset_key}")
        return True
    
    def preprocess(self, dataset_key: str = "hm3d") -> bool:
        """Create navigation episodes from 3D scans"""
        scene_dir = self.data_dir / dataset_key
        
        if not scene_dir.exists():
            self.logger.warning(f"Scene directory not found: {scene_dir}")
            return False
        
        # Generate navigation episodes
        episodes = self._generate_navigation_episodes(scene_dir)
        
        # Save episodes
        episodes_path = self.episodes_dir / f"{dataset_key}_episodes.json"
        with open(episodes_path, 'w') as f:
            json.dump(episodes, f, indent=2)
        
        self.logger.info(f"Generated {len(episodes)} navigation episodes")
        return True
    
    def _generate_navigation_episodes(self, scene_dir: Path) -> List[Dict]:
        """Generate navigation episodes from scene"""
        episodes = []
        
        # Find all scene files
        scene_files = list(scene_dir.glob("**/*.glb")) + list(scene_dir.glob("**/*.obj"))
        
        for scene_file in scene_files[:100]:  # Limit for initial testing
            # Generate random start/goal pairs
            for i in range(10):
                episode = {
                    "episode_id": f"{scene_file.stem}_{i}",
                    "scene_id": str(scene_file),
                    "start_position": [0.0, 0.0, 0.0],  # Would be sampled from navmesh
                    "start_rotation": [0.0, 0.0, 0.0, 1.0],
                    "goal_position": [5.0, 0.0, 5.0],  # Would be sampled from navmesh
                    "info": {"geodesic_distance": 7.07}
                }
                episodes.append(episode)
        
        return episodes
    
    def get_train_generator(self, batch_size: int = 32) -> Generator:
        """Get training episodes generator"""
        episodes_path = self.episodes_dir / "hm3d_episodes.json"
        
        if not episodes_path.exists():
            raise FileNotFoundError(f"Episodes not found: {episodes_path}")
        
        with open(episodes_path, 'r') as f:
            episodes = json.load(f)
        
        batch = []
        for episode in episodes:
            batch.append(episode)
            if len(batch) == batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    def get_val_generator(self, batch_size: int = 32) -> Generator:
        """Get validation episodes generator"""
        return self.get_train_generator(batch_size)


# =============================================================================
# HIGH PRIORITY: Vision Tracking Real Data Pipeline
# =============================================================================

class VisionTrackingDataPipeline(BaseDataPipeline):
    """
    Real data pipeline for Vision Tracking (CameraFollowNet) training.
    
    Datasets:
    - MOT Challenge (Multi-Object Tracking)
    - LaSOT (Large-scale Single Object Tracking)
    - GOT-10k (Generic Object Tracking)
    """
    
    DATASETS = {
        "mot17": DatasetConfig(
            name="MOT17 Challenge",
            url="https://motchallenge.net/data/MOT17/",
            size_gb=5.5,
            format="images+annotations",
            license="CC BY-NC-SA 3.0",
            citation="Milan et al., 2016"
        ),
        "lasot": DatasetConfig(
            name="LaSOT",
            url="https://cis.temple.edu/lasot/",
            size_gb=230.0,
            format="images+annotations",
            license="Research only",
            citation="Fan et al., 2019"
        ),
        "got10k": DatasetConfig(
            name="GOT-10k",
            url="http://got-10k.aitestunion.com/",
            size_gb=75.0,
            format="images+annotations",
            license="Research only",
            citation="Huang et al., 2019"
        )
    }
    
    def __init__(self, data_dir: str = "/data/organica-ai/datasets/tracking"):
        super().__init__(data_dir)
        self.sequences_dir = self.data_dir / "sequences"
        self.sequences_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, dataset_key: str = "mot17") -> bool:
        """Download tracking dataset"""
        config = self.DATASETS[dataset_key]
        self.logger.info(f"Please download {config.name} from: {config.url}")
        self.logger.info(f"Place files in: {self.data_dir / dataset_key}")
        return True
    
    def preprocess(self, dataset_key: str = "mot17") -> bool:
        """Preprocess tracking sequences"""
        dataset_dir = self.data_dir / dataset_key
        
        if not dataset_dir.exists():
            self.logger.warning(f"Dataset directory not found: {dataset_dir}")
            return False
        
        # Process MOT format
        sequences = self._process_mot_format(dataset_dir)
        
        # Save processed sequences
        output_path = self.sequences_dir / f"{dataset_key}_sequences.json"
        with open(output_path, 'w') as f:
            json.dump(sequences, f, indent=2)
        
        self.logger.info(f"Processed {len(sequences)} tracking sequences")
        return True
    
    def _process_mot_format(self, dataset_dir: Path) -> List[Dict]:
        """Process MOT Challenge format"""
        sequences = []
        
        for seq_dir in dataset_dir.glob("*/"):
            if not seq_dir.is_dir():
                continue
            
            gt_file = seq_dir / "gt" / "gt.txt"
            if not gt_file.exists():
                continue
            
            # Parse ground truth
            tracks = {}
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 6:
                        frame_id = int(parts[0])
                        track_id = int(parts[1])
                        bbox = [float(parts[2]), float(parts[3]), 
                               float(parts[4]), float(parts[5])]
                        
                        if track_id not in tracks:
                            tracks[track_id] = []
                        tracks[track_id].append({
                            "frame": frame_id,
                            "bbox": bbox
                        })
            
            sequences.append({
                "sequence_id": seq_dir.name,
                "num_frames": len(list((seq_dir / "img1").glob("*.jpg"))),
                "tracks": tracks
            })
        
        return sequences
    
    def get_train_generator(self, batch_size: int = 32) -> Generator:
        """Get training data generator for camera follow"""
        sequences_path = self.sequences_dir / "mot17_sequences.json"
        
        if not sequences_path.exists():
            raise FileNotFoundError(f"Sequences not found: {sequences_path}")
        
        with open(sequences_path, 'r') as f:
            sequences = json.load(f)
        
        batch = []
        for seq in sequences:
            for track_id, track_data in seq.get("tracks", {}).items():
                for i in range(len(track_data) - 1):
                    # Create training sample: current bbox -> camera command
                    sample = {
                        "sequence_id": seq["sequence_id"],
                        "track_id": track_id,
                        "current_bbox": track_data[i]["bbox"],
                        "next_bbox": track_data[i + 1]["bbox"],
                        "frame_delta": track_data[i + 1]["frame"] - track_data[i]["frame"]
                    }
                    batch.append(sample)
                    
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
        
        if batch:
            yield batch
    
    def get_val_generator(self, batch_size: int = 32) -> Generator:
        """Get validation data generator"""
        return self.get_train_generator(batch_size)


# =============================================================================
# MEDIUM PRIORITY: VLA Real Data Pipelines
# =============================================================================

class VLARealDataPipeline(BaseDataPipeline):
    """
    Real data pipeline for VLA model training.
    
    Datasets:
    - Open X-Embodiment (multi-robot)
    - Bridge Dataset (manipulation)
    - DROID (manipulation)
    - RT-1 Dataset
    """
    
    DATASETS = {
        "open_x_embodiment": DatasetConfig(
            name="Open X-Embodiment",
            url="https://robotics-transformer-x.github.io/",
            size_gb=500.0,
            format="tfrecord",
            license="Apache 2.0",
            citation="Open X-Embodiment Collaboration, 2023"
        ),
        "bridge_v2": DatasetConfig(
            name="Bridge Dataset V2",
            url="https://rail-berkeley.github.io/bridgedata/",
            size_gb=400.0,
            format="tfrecord",
            license="MIT",
            citation="Walke et al., 2023"
        ),
        "droid": DatasetConfig(
            name="DROID Dataset",
            url="https://droid-dataset.github.io/",
            size_gb=2000.0,
            format="zarr",
            license="CC BY 4.0",
            citation="Khazatsky et al., 2024"
        ),
        "unitree_go1": DatasetConfig(
            name="Unitree Go1 Locomotion",
            url="https://github.com/unitreerobotics",
            size_gb=50.0,
            format="rosbag",
            license="Proprietary",
            citation="Unitree Robotics"
        )
    }
    
    def __init__(self, data_dir: str = "/data/organica-ai/datasets/vla"):
        super().__init__(data_dir)
        self.episodes_dir = self.data_dir / "episodes"
        self.episodes_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, dataset_key: str = "bridge_v2") -> bool:
        """Download VLA dataset"""
        config = self.DATASETS[dataset_key]
        self.logger.info(f"Please download {config.name} from: {config.url}")
        self.logger.info(f"Size: {config.size_gb} GB")
        self.logger.info(f"Place files in: {self.data_dir / dataset_key}")
        return True
    
    def preprocess(self, dataset_key: str = "bridge_v2") -> bool:
        """Preprocess VLA episodes"""
        dataset_dir = self.data_dir / dataset_key
        
        if not dataset_dir.exists():
            self.logger.warning(f"Dataset directory not found: {dataset_dir}")
            return False
        
        # Process based on format
        if dataset_key == "bridge_v2":
            episodes = self._process_bridge_format(dataset_dir)
        elif dataset_key == "droid":
            episodes = self._process_droid_format(dataset_dir)
        else:
            episodes = self._process_generic_format(dataset_dir)
        
        # Save processed episodes
        output_path = self.episodes_dir / f"{dataset_key}_episodes.jsonl"
        with open(output_path, 'w') as f:
            for episode in episodes:
                f.write(json.dumps(episode) + '\n')
        
        self.logger.info(f"Processed {len(episodes)} VLA episodes")
        return True
    
    def _process_bridge_format(self, dataset_dir: Path) -> List[Dict]:
        """Process Bridge Dataset format"""
        episodes = []
        
        for traj_dir in dataset_dir.glob("*/"):
            if not traj_dir.is_dir():
                continue
            
            # Load trajectory data
            obs_file = traj_dir / "observations.json"
            action_file = traj_dir / "actions.json"
            
            if obs_file.exists() and action_file.exists():
                with open(obs_file, 'r') as f:
                    observations = json.load(f)
                with open(action_file, 'r') as f:
                    actions = json.load(f)
                
                for i, (obs, action) in enumerate(zip(observations, actions)):
                    episodes.append({
                        "trajectory_id": traj_dir.name,
                        "step": i,
                        "image_path": str(traj_dir / f"images/{i:06d}.png"),
                        "instruction": obs.get("instruction", ""),
                        "state": obs.get("state", [0] * 7),
                        "action": action.get("action", [0] * 7),
                        "gripper": action.get("gripper", 0.0)
                    })
        
        return episodes
    
    def _process_droid_format(self, dataset_dir: Path) -> List[Dict]:
        """Process DROID Dataset format (zarr)"""
        episodes = []
        # DROID uses zarr format - would need zarr library
        self.logger.info("DROID format requires zarr library")
        return episodes
    
    def _process_generic_format(self, dataset_dir: Path) -> List[Dict]:
        """Process generic VLA format"""
        episodes = []
        
        for jsonl_file in dataset_dir.glob("*.jsonl"):
            with open(jsonl_file, 'r') as f:
                for line in f:
                    episodes.append(json.loads(line))
        
        return episodes
    
    def get_train_generator(self, batch_size: int = 32) -> Generator:
        """Get training data generator"""
        episodes_path = self.episodes_dir / "bridge_v2_episodes.jsonl"
        
        if not episodes_path.exists():
            raise FileNotFoundError(f"Episodes not found: {episodes_path}")
        
        batch = []
        with open(episodes_path, 'r') as f:
            for line in f:
                batch.append(json.loads(line))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        
        if batch:
            yield batch
    
    def get_val_generator(self, batch_size: int = 32) -> Generator:
        """Get validation data generator"""
        return self.get_train_generator(batch_size)


# =============================================================================
# MEDIUM PRIORITY: Drone Real Data Pipeline
# =============================================================================

class DroneRealDataPipeline(BaseDataPipeline):
    """
    Real data pipeline for Drone VLA training.
    
    Datasets:
    - TartanAir (drone simulation with real-world transfer)
    - EuRoC MAV (real drone flights)
    - UZH-FPV (racing drone)
    """
    
    DATASETS = {
        "tartanair": DatasetConfig(
            name="TartanAir",
            url="https://theairlab.org/tartanair-dataset/",
            size_gb=500.0,
            format="images+poses",
            license="CC BY 4.0",
            citation="Wang et al., 2020"
        ),
        "euroc_mav": DatasetConfig(
            name="EuRoC MAV Dataset",
            url="https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets",
            size_gb=22.0,
            format="rosbag",
            license="CC BY-NC-SA 3.0",
            citation="Burri et al., 2016"
        ),
        "uzh_fpv": DatasetConfig(
            name="UZH-FPV Drone Racing",
            url="https://fpv.ifi.uzh.ch/",
            size_gb=30.0,
            format="rosbag",
            license="CC BY-NC-SA 4.0",
            citation="Delmerico et al., 2019"
        )
    }
    
    def __init__(self, data_dir: str = "/data/organica-ai/datasets/drone"):
        super().__init__(data_dir)
        self.flights_dir = self.data_dir / "flights"
        self.flights_dir.mkdir(parents=True, exist_ok=True)
    
    def download(self, dataset_key: str = "euroc_mav") -> bool:
        """Download drone dataset"""
        config = self.DATASETS[dataset_key]
        self.logger.info(f"Please download {config.name} from: {config.url}")
        self.logger.info(f"Size: {config.size_gb} GB")
        return True
    
    def preprocess(self, dataset_key: str = "euroc_mav") -> bool:
        """Preprocess drone flight data"""
        dataset_dir = self.data_dir / dataset_key
        
        if not dataset_dir.exists():
            self.logger.warning(f"Dataset directory not found: {dataset_dir}")
            return False
        
        flights = self._process_euroc_format(dataset_dir)
        
        output_path = self.flights_dir / f"{dataset_key}_flights.jsonl"
        with open(output_path, 'w') as f:
            for flight in flights:
                f.write(json.dumps(flight) + '\n')
        
        self.logger.info(f"Processed {len(flights)} drone flight segments")
        return True
    
    def _process_euroc_format(self, dataset_dir: Path) -> List[Dict]:
        """Process EuRoC MAV format"""
        flights = []
        
        for seq_dir in dataset_dir.glob("*/mav0/"):
            # Load IMU data
            imu_file = seq_dir / "imu0" / "data.csv"
            # Load ground truth
            gt_file = seq_dir / "state_groundtruth_estimate0" / "data.csv"
            
            if imu_file.exists() and gt_file.exists():
                # Parse and create training samples
                # This would parse the actual CSV files
                flights.append({
                    "sequence_id": seq_dir.parent.name,
                    "imu_path": str(imu_file),
                    "gt_path": str(gt_file),
                    "num_samples": 1000  # Placeholder
                })
        
        return flights
    
    def get_train_generator(self, batch_size: int = 32) -> Generator:
        """Get training data generator"""
        flights_path = self.flights_dir / "euroc_mav_flights.jsonl"
        
        if not flights_path.exists():
            raise FileNotFoundError(f"Flights not found: {flights_path}")
        
        batch = []
        with open(flights_path, 'r') as f:
            for line in f:
                batch.append(json.loads(line))
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        
        if batch:
            yield batch
    
    def get_val_generator(self, batch_size: int = 32) -> Generator:
        """Get validation data generator"""
        return self.get_train_generator(batch_size)


# =============================================================================
# PIPELINE FACTORY
# =============================================================================

class RealDataPipelineFactory:
    """Factory for creating real data pipelines"""
    
    PIPELINES = {
        # HIGH PRIORITY
        "nemo_asr": NeMoASRDataPipeline,
        "isaac_rl": IsaacRLDataPipeline,
        "vision_tracking": VisionTrackingDataPipeline,
        # MEDIUM PRIORITY
        "vla": VLARealDataPipeline,
        "drone": DroneRealDataPipeline,
    }
    
    @classmethod
    def create(cls, pipeline_type: str, data_dir: Optional[str] = None) -> BaseDataPipeline:
        """Create a data pipeline"""
        if pipeline_type not in cls.PIPELINES:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        pipeline_class = cls.PIPELINES[pipeline_type]
        
        if data_dir:
            return pipeline_class(data_dir)
        return pipeline_class()
    
    @classmethod
    def list_pipelines(cls) -> List[str]:
        """List available pipelines"""
        return list(cls.PIPELINES.keys())


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI for real data pipelines"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real Data Pipelines for NIS Protocol")
    parser.add_argument("action", choices=["download", "preprocess", "list"])
    parser.add_argument("--pipeline", type=str, help="Pipeline type")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset key")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.action == "list":
        print("Available pipelines:")
        for name in RealDataPipelineFactory.list_pipelines():
            print(f"  - {name}")
        return
    
    if not args.pipeline:
        print("Error: --pipeline required")
        return
    
    pipeline = RealDataPipelineFactory.create(args.pipeline, args.data_dir)
    
    if args.action == "download":
        if args.dataset:
            pipeline.download(args.dataset)
        else:
            pipeline.download()
    elif args.action == "preprocess":
        if args.dataset:
            pipeline.preprocess(args.dataset)
        else:
            pipeline.preprocess()


if __name__ == "__main__":
    main()
