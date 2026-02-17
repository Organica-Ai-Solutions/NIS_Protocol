# Real Data Training Guide for NIS Protocol

**Created:** January 31, 2026  
**Purpose:** Guide for retraining NIS Protocol models with real-world data

---

## 📊 Priority Overview

### 🔴 HIGH PRIORITY (Production Critical)

| Model | Current Data | Real Data Source | Size | Status |
|-------|--------------|------------------|------|--------|
| **NeMo ASR** | Synthetic audio | LibriSpeech, Common Voice | 60-100 GB | Ready |
| **Isaac RL Navigation** | Isaac Sim | HM3D, Gibson, MP3D | 130-270 GB | Ready |
| **Vision Tracking** | Synthetic | MOT17, LaSOT, GOT-10k | 5-230 GB | Ready |
| **VLA-Navigation** | Isaac Sim | HM3D real scans | 130 GB | Training |
| **VLA-Drone** | PX4 SITL | EuRoC MAV, TartanAir | 22-500 GB | Training |

### 🟡 MEDIUM PRIORITY (Enhancement)

| Model | Current Data | Real Data Source | Size | Status |
|-------|--------------|------------------|------|--------|
| **VLA-Quadruped** | MuJoCo | Unitree Go1/Go2 logs | 50 GB | Training |
| **VLA-Bimanual** | Synthetic | ALOHA real demos | 100 GB | Pending |
| **VLA-Mobile** | Synthetic | TurtleBot/Fetch logs | 30 GB | Pending |
| **VLA-Manipulation** | Synthetic | Bridge V2, DROID | 400-2000 GB | Pending |

---

## 🛠️ Data Pipeline Scripts

### Location
```
scripts/training/
├── real_data_pipelines.py      # Data pipeline classes
├── test_real_data_training.py  # Test scripts
└── train_with_real_data.sh     # H100 training script
```

### Available Pipelines

```python
from scripts.training.real_data_pipelines import RealDataPipelineFactory

# List all pipelines
pipelines = RealDataPipelineFactory.list_pipelines()
# ['nemo_asr', 'isaac_rl', 'vision_tracking', 'vla', 'drone']

# Create a pipeline
pipeline = RealDataPipelineFactory.create("nemo_asr")
pipeline.download("librispeech_clean_100")
pipeline.preprocess("librispeech_clean_100")
```

---

## 📥 Dataset Download Instructions

### HIGH PRIORITY Datasets

#### 1. LibriSpeech (NeMo ASR)
```bash
# Download clean-100 (6.3 GB)
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz

# Download clean-360 (23 GB)
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz

# Extract
tar -xzf train-clean-100.tar.gz -C /data/organica-ai/datasets/asr/
```

#### 2. HM3D (Isaac RL Navigation)
```bash
# Requires registration at:
# https://github.com/facebookresearch/habitat-matterport3d-dataset

# After approval, download scenes
python -m habitat_sim.utils.datasets_download --uids hm3d
```

#### 3. MOT17 (Vision Tracking)
```bash
# Download from MOT Challenge
wget https://motchallenge.net/data/MOT17.zip
unzip MOT17.zip -d /data/organica-ai/datasets/tracking/
```

### MEDIUM PRIORITY Datasets

#### 4. Bridge V2 (VLA Manipulation)
```bash
# Download from:
# https://rail-berkeley.github.io/bridgedata/

# Use gsutil for Google Cloud Storage
gsutil -m cp -r gs://rail-berkeley-bridge-data/v2/ /data/organica-ai/datasets/vla/bridge_v2/
```

#### 5. EuRoC MAV (Drone)
```bash
# Download from ASL datasets
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip
unzip MH_01_easy.zip -d /data/organica-ai/datasets/drone/euroc_mav/
```

---

## 🚀 Training Commands

### Run on H100 Cluster

```bash
# SSH to H100
ssh nvidia@awesome-gpu-name

# HIGH PRIORITY training on GPUs 2,3,5
./train_with_real_data.sh high 2,3,5

# MEDIUM PRIORITY training
./train_with_real_data.sh medium 2,3,5

# ALL training (sequential)
./train_with_real_data.sh all 2,3,5
```

### Individual Model Training

```bash
# NeMo ASR with LibriSpeech
CUDA_VISIBLE_DEVICES=2 python3 -c "
from scripts.training.real_data_pipelines import NeMoASRDataPipeline
pipeline = NeMoASRDataPipeline()
# Training code here
"

# Vision Tracking with MOT17
CUDA_VISIBLE_DEVICES=3 python3 -c "
from scripts.training.real_data_pipelines import VisionTrackingDataPipeline
pipeline = VisionTrackingDataPipeline()
# Training code here
"
```

---

## 🧪 Testing

### Run Tests

```bash
# All tests
python scripts/training/test_real_data_training.py

# High priority only
python scripts/training/test_real_data_training.py --priority high

# Medium priority only
python scripts/training/test_real_data_training.py --priority medium

# Verbose
python scripts/training/test_real_data_training.py -v
```

### Expected Test Results

```
HIGH PRIORITY:
✅ TestNeMoASRPipeline (4 tests)
✅ TestIsaacRLPipeline (3 tests)
✅ TestVisionTrackingPipeline (3 tests)
✅ TestNeMoASRTraining (2 tests)
✅ TestVisionTrackingTraining (2 tests)

MEDIUM PRIORITY:
✅ TestVLARealDataPipeline (3 tests)
✅ TestDroneRealDataPipeline (2 tests)
✅ TestVLATraining (3 tests)
```

---

## 📁 Output Locations

### Models
```
/data/organica-ai/models/
├── nemo_asr_realdata.pt           # NeMo ASR with LibriSpeech
├── vision_tracking_realdata.pt    # Vision Tracking with MOT17
├── isaac_rl_realdata.pt           # Isaac RL with HM3D
├── vla_realdata_navigation/       # VLA Navigation
├── vla_realdata_drone/            # VLA Drone
├── vla_realdata_quadruped/        # VLA Quadruped
├── vla_realdata_bimanual/         # VLA Bimanual
└── vla_realdata_mobile/           # VLA Mobile
```

### Logs
```
/data/organica-ai/logs/
├── nemo_asr_realdata.log
├── vision_tracking_realdata.log
├── isaac_rl_realdata.log
├── vla_nav_realdata.log
├── vla_drone_realdata.log
└── vla_quad_realdata.log
```

---

## 📈 Training Progress Monitoring

```bash
# Check GPU status
nvidia-smi

# Watch training logs
tail -f /data/organica-ai/logs/vla_nav_realdata.log

# Check all active training
tmux ls
tmux attach -t vla_nav_realdata
```

---

## ⚠️ Important Notes

1. **Data Licensing**: Some datasets require registration and license agreement
2. **Storage**: Ensure sufficient disk space (500GB+ recommended)
3. **GPU Memory**: H100 80GB can handle all models
4. **Training Time**: 
   - HIGH PRIORITY: 4-8 hours per model
   - MEDIUM PRIORITY: 8-12 hours per model

---

## 🔗 Dataset References

| Dataset | Paper | URL |
|---------|-------|-----|
| LibriSpeech | Panayotov et al., 2015 | openslr.org/12 |
| HM3D | Ramakrishnan et al., 2021 | github.com/facebookresearch/habitat-matterport3d-dataset |
| MOT17 | Milan et al., 2016 | motchallenge.net |
| Bridge V2 | Walke et al., 2023 | rail-berkeley.github.io/bridgedata |
| EuRoC MAV | Burri et al., 2016 | projects.asl.ethz.ch/datasets |
| Open X-Embodiment | Collaboration, 2023 | robotics-transformer-x.github.io |
