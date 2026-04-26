"""
GATE 4: NVIDIA COSMOS SIM-TO-REAL PIPELINE
==========================================
Architecture (from NVIDIA official docs, Feb 2026):

  Pi 5 (robot edge):
    - Hiwonder ROS 2 stack (servo control, camera logging)
    - ROS 2 bags for deterministic replay/debug
    - ONNX Runtime CPU for lightweight inference

  GPU Workstation (offline training):
    - Isaac Sim + CosmosWriter: generate RGB/depth/seg/edges
    - Cosmos Transfer2.5: sim-to-photo augmentation (needs 65.4 GB VRAM)
    - Cosmos Predict2-2B: action-conditioned future prediction (needs 32.54 GB VRAM)
    - Training: DP3 / Diffusion Policy / Cosmos Policy post-training
    - Export: ONNX + TensorRT for Pi deployment

VRAM REQUIREMENTS (official NVIDIA Cosmos docs, Feb 18 2026):
  Transfer2.5-2B:         65.4 GB VRAM  (H100 80GB class)
  Predict2-2B-Video2World: 32.54 GB VRAM
  Predict2-2B (at 480p):  ~26 GB VRAM
  Predict2-14B:           56.38 GB VRAM

COSMOS POLICY (from arXiv 2601.16163, Jan 22 2026):
  - NOT plug-and-play for Hiwonder (trained for ALOHA 2 bimanual setup)
  - Use as METHOD blueprint: latent frame injection + action conditioning
  - Requires: top-down + 2 wrist cameras, 25 Hz control, ALOHA hardware
  - Our setup: 1 top-down camera -> MUST retrain from scratch post-training

WORKFLOW:
  1. Collect demonstrations (pi_data_collector.py runs on Pi)
  2. Isaac Sim -> import xarm_ai.urdf -> set up cameras
  3. CosmosWriter -> export RGB/depth/seg/edges clips
  4. Cosmos Transfer2.5 -> photorealize sim clips
  5. Train perception/policy on augmented data
  6. Export ONNX -> deploy to Pi

Usage:
  python cosmos_pipeline.py --step collect     # collect demo data on Pi
  python cosmos_pipeline.py --step sim         # validate Isaac Sim setup
  python cosmos_pipeline.py --step transfer    # run Transfer2.5 augmentation
  python cosmos_pipeline.py --step train       # train policy
  python cosmos_pipeline.py --step export      # export ONNX for Pi
  python cosmos_pipeline.py --step deploy      # validate on Pi
  python cosmos_pipeline.py --status           # show pipeline status
"""

import json, math, time, sys, os, argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--step', choices=['collect','sim','transfer','train','export','deploy'])
parser.add_argument('--status', action='store_true')
parser.add_argument('--demos', type=int, default=50)
args = parser.parse_args()

PI     = 'http://192.168.1.163:8085'
CALIB  = Path('data/calib_results.json')
CAMCAL = Path('data/camera_cal.json')
DATA   = Path('data/cosmos')
DATA.mkdir(parents=True, exist_ok=True)
SEP    = '=' * 70

import urllib.request as _ur

def get(p, t=15):
    try: return json.loads(_ur.urlopen(PI+p, timeout=t).read())
    except Exception as e: return {}

def post(p, b=None, t=20):
    try:
        d = json.dumps(b or {}).encode()
        r = _ur.Request(PI+p, data=d, headers={'Content-Type':'application/json'})
        return json.loads(_ur.urlopen(r, timeout=t).read())
    except Exception as e: return {}


# ============================================================================
# STATUS - show current pipeline state
# ============================================================================

if args.status or not args.step:
    print(SEP)
    print('  COSMOS SIM-TO-REAL PIPELINE STATUS')
    print(SEP)

    checks = {
        'Gate 1: Servo deviation':       Path('data/calib_results.json'),
        'Gate 2: Camera intrinsics':      Path('data/camera_intrinsics.json'),
        'Gate 3: URDF':                  Path('xarm_ai.urdf'),
        'Camera calibration (vision)':   Path('data/camera_cal.json'),
        'Demo data collected':            DATA / 'demos',
        'Isaac Sim config':               DATA / 'isaac_sim_config.json',
        'CosmosWriter clips':             DATA / 'sim_clips',
        'Transfer augmented':             DATA / 'transfer_clips',
        'Trained model':                  DATA / 'model',
        'ONNX export':                    DATA / 'model/xarm_policy.onnx',
    }

    for label, p in checks.items():
        if p.exists():
            if p.is_file():
                sz = p.stat().st_size
                print(f'  [DONE]    {label:<40} ({sz:,} bytes)')
            else:
                items = list(p.glob('*')) if p.is_dir() else []
                print(f'  [DONE]    {label:<40} ({len(items)} items)')
        else:
            print(f'  [MISSING] {label}')

    print()
    print('  VRAM guide (official NVIDIA docs, Feb 2026):')
    print('    Cosmos Transfer2.5-2B:     65.4 GB  H100-80GB / A100-80GB')
    print('    Cosmos Predict2-2B:        32.5 GB  RTX 4090 x2 or A6000')
    print('    Cosmos Predict2-2B 480p:  ~26.0 GB  RTX 4090 single')
    print('    Cosmos Predict2-14B:       56.4 GB  H100')
    print()
    print('  Deployment (Pi 5):')
    print('    ONNX Runtime CPU (lightweight policy):  ~4 GB RAM')
    print('    Triton remote GPU inference:  <100ms latency on LAN')
    print()
    print('  Run steps in order:')
    for step in ['collect', 'sim', 'transfer', 'train', 'export', 'deploy']:
        print(f'    python cosmos_pipeline.py --step {step}')
    sys.exit(0)


# ============================================================================
# STEP: COLLECT - gather demonstration data from Pi
# ============================================================================

if args.step == 'collect':
    print(SEP)
    print('  COLLECT: Demonstration Data via Pi Agent')
    print(SEP)

    DEMOS_DIR = DATA / 'demos'
    DEMOS_DIR.mkdir(exist_ok=True)

    calib = json.loads(CALIB.read_text()) if CALIB.exists() else {}

    S3_PICK = calib.get('s3_pick', 177)
    S4_PICK = calib.get('s4_pick', 814)
    S5_PICK = calib.get('s5_pick', 432)
    S6_CAL  = calib.get('s6_pick', 400)
    PLACE_JOINTS = calib.get('place_joints', {})

    HOME  = {'1':100,'2':500,'3':310,'4':870,'5':680,'6':500}
    PICK  = {'1':100,'2':500,'3':S3_PICK,'4':S4_PICK,'5':S5_PICK,'6':S6_CAL}
    GRIP  = {'1':500,'2':500,'3':S3_PICK,'4':S4_PICK,'5':S5_PICK,'6':S6_CAL}
    LIFT  = {**HOME,'1':500,'6':S6_CAL}
    pj    = PLACE_JOINTS.get('left90', {'3':220,'4':827,'5':425,'6':875})
    PLACE = {**{str(k):int(v) for k,v in pj.items()},'1':500,'2':500}
    RELEASE={**{str(k):int(v) for k,v in pj.items()},'1':100,'2':500}

    def move(svs, ms=1000, label=''):
        r = post('/arm/group_move', {'positions': svs, 'duration_ms': ms})
        sim = r.get('simulation', False)
        s   = ' '.join(f'S{k}={v}' for k,v in sorted(svs.items()))
        print(f'  [{label or "MOVE"}] {s}  [{"SIM" if sim else "OK"}]')
        time.sleep(ms/1000.0 + 0.3)
        return r

    import base64

    def snap_and_log(episode, step_name):
        d    = get('/camera/snapshot', t=20)
        img  = d.get('image_base64') or d.get('image')
        if img:
            p = DEMOS_DIR / f'ep{episode:04d}_{step_name}.jpg'
            p.write_bytes(base64.b64decode(img))
        return img

    def log_episode(episode, steps):
        record = {
            'episode':    episode,
            'timestamp':  time.strftime('%Y-%m-%dT%H:%M:%S'),
            'place':      'left90',
            'steps':      steps,
            'calibration': {'s3': S3_PICK, 's4': S4_PICK, 's5': S5_PICK, 's6': S6_CAL},
        }
        p = DEMOS_DIR / f'ep{episode:04d}_log.json'
        p.write_text(json.dumps(record, indent=2))
        return p

    print(f'  Collecting {args.demos} pick-and-place demonstrations')
    print(f'  Data dir: {DEMOS_DIR}')
    print()

    for ep in range(args.demos):
        print(f'  --- EPISODE {ep+1}/{args.demos} ---')
        steps = []

        pose_seq = [
            (HOME,    1500, 'home'),
            (PICK,     900, 'pick_down'),
            (GRIP,     400, 'grip'),
            (LIFT,    1000, 'lift'),
            ({'1':500,'2':500,'3':310,'4':870,'5':680,'6':875}, 1500, 'rotate'),
            (PLACE,   1000, 'place_down'),
            (RELEASE,  500, 'release'),
            (HOME,    1500, 'home_end'),
        ]

        for (pose, ms, label) in pose_seq:
            move(pose, ms=ms, label=label)
            snap_and_log(ep, label)
            arm_st = get('/arm/status', t=8)
            steps.append({
                'label':     label,
                'commanded': pose,
                'actual':    arm_st.get('positions', {}),
                'timestamp': time.time(),
            })

        log_episode(ep, steps)
        print(f'  Episode {ep+1} saved\n')

        if ep < args.demos - 1:
            print('  Place lighter back at pick zone...')
            time.sleep(4.0)

    print(f'  Collected {args.demos} episodes in {DEMOS_DIR}')
    sys.exit(0)


# ============================================================================
# STEP: SIM - Isaac Sim setup guide
# ============================================================================

if args.step == 'sim':
    print(SEP)
    print('  ISAAC SIM SETUP GUIDE (from NVIDIA docs, Isaac Sim 5.1.x GA)')
    print(SEP)

    # Load calibration data for camera setup
    cam_cal = json.loads(CAMCAL.read_text()) if CAMCAL.exists() else {}
    intr    = Path('data/camera_intrinsics.json')
    intr_d  = json.loads(intr.read_text()) if intr.exists() else {}

    fx = intr_d.get('focal_length_px', [1024.0, 1024.0])[0]
    fy = intr_d.get('focal_length_px', [1024.0, 1024.0])[1]
    cx = intr_d.get('principal_point', [640.0, 360.0])[0]
    cy = intr_d.get('principal_point', [640.0, 360.0])[1]
    w  = intr_d.get('resolution', [1280, 720])[0]
    h  = intr_d.get('resolution', [1280, 720])[1]
    H  = cam_cal.get('estimated_height_cm', 60.0)

    cfg = {
        'urdf_path':       str(Path('xarm_ai.urdf').absolute()),
        'camera': {
            'width': w, 'height': h,
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
            'mount_height_m': round(H / 100.0, 3),
            'type': 'top_down',
            'ros2_topic': '/xarm/camera/image_raw',
        },
        'joint_names':    ['joint6','joint5','joint4','joint3','joint2','joint1'],
        'servo_range':    {'min': 0, 'max': 1000, 'neutral': 500},
        'deg_per_unit':   0.24,
        'link_lengths_cm': {'L1': 6.9, 'L2': 9.5, 'L3': 9.5, 'L4': 16.9},
        'cosmos_writer': {
            'modalities':  ['rgb', 'depth', 'segmentation', 'edges'],
            'fps':         25,
            'resolution':  [w, h],
            'output_dir':  str(DATA / 'sim_clips'),
        },
        'generated_at':   time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    cfg_path = DATA / 'isaac_sim_config.json'
    cfg_path.write_text(json.dumps(cfg, indent=2))

    print()
    print('  URDF Import (isaacsim.asset.importer.urdf):')
    print(f'    File: {cfg["urdf_path"]}')
    print('    Note: Isaac Sim replaces special chars with underscores')
    print('    -> Fix names in URDF if import warnings appear')
    print()
    print('  Camera Setup (isaacsim.sensors.rtx.placement):')
    print(f'    Type: pinhole  Width: {w}  Height: {h}')
    print(f'    fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}')
    print(f'    Height above table: {H:.1f} cm')
    print(f'    Mount: top-down (camera z-axis pointing down)')
    print()
    print('  Camera calibration export:')
    print('    Extension: isaacsim.replicator.agent.camera_calibration')
    print('    -> saves calibration JSON (direction/location/FOV polygon)')
    print('    -> compare with data/camera_intrinsics.json for sim-real gap')
    print()
    print('  CosmosWriter setup:')
    print('    Extension: isaacsim.replicator.cosmos (CosmosWriter)')
    print('    Modalities:', cfg['cosmos_writer']['modalities'])
    print('    Output:', cfg['cosmos_writer']['output_dir'])
    print()
    print('  WARNING: Isaac Sim 6.0 is "Early Developer Release" (incomplete docs).')
    print('           Use Isaac Sim 5.1.x (GA) for stable CosmosWriter support.')
    print()
    print(f'  Config saved: {cfg_path}')
    sys.exit(0)


# ============================================================================
# STEP: TRANSFER - Cosmos Transfer2.5 augmentation
# ============================================================================

if args.step == 'transfer':
    print(SEP)
    print('  COSMOS TRANSFER2.5 -- SIM-TO-PHOTO AUGMENTATION')
    print(SEP)
    print()
    print('  VRAM REQUIREMENT: 65.4 GB (Transfer2.5-2B)')
    print('  TARGET GPU:       NVIDIA H100-80GB or A100-80GB')
    print()

    clips_dir = DATA / 'sim_clips'
    out_dir   = DATA / 'transfer_clips'
    out_dir.mkdir(exist_ok=True)

    clips = list(clips_dir.glob('*.mp4')) if clips_dir.exists() else []
    print(f'  Found {len(clips)} sim clips in {clips_dir}')
    print()

    print('  SETUP COMMANDS (run on GPU workstation):')
    print()
    print('  # 1. Install Cosmos Transfer2.5')
    print('  git clone https://github.com/nvidia-cosmos/cosmos-transfer2.5')
    print('  cd cosmos-transfer2.5')
    print('  pip install -e .')
    print()
    print('  # 2. Download 2B model (needs HF token)')
    print('  huggingface-cli login')
    print('  python scripts/download_model.py --model Transfer2.5-2B')
    print()
    print('  # 3. Create parameter file for multi-controlnet (sim-to-real)')
    params = {
        'model': 'cosmos-transfer2.5-2b',
        'input_dir': str(clips_dir.absolute()),
        'output_dir': str(out_dir.absolute()),
        'controlnets': ['depth', 'segmentation', 'edges'],
        'prompt': (
            'robotic arm gripper picking up a small object from a table, '
            'bright indoor lighting, photorealistic, high quality'
        ),
        'resolution': '720p',
        'fps': 25,
        'num_steps': 35,
        'guidance_scale': 5.0,
    }
    params_path = DATA / 'transfer_params.json'
    params_path.write_text(json.dumps(params, indent=2))
    print(f'  # Params saved: {params_path}')
    print()
    print('  # 4. Run inference')
    print(f'  python examples/inference.py --params_file {params_path}')
    print()
    print('  # 5. Verify outputs')
    print('  - Transfer videos should preserve robot motion')
    print('  - Visually check that gripper tip is still identifiable')
    print('  - Compare object positions between sim and transfer frames')
    print()
    print('  NOTE: 720p/16FPS/5s on H100 PCIe: ~79 sec per clip')
    print('        Budget for ~100 clips: ~130 minutes on H100')
    sys.exit(0)


# ============================================================================
# STEP: TRAIN - Policy training
# ============================================================================

if args.step == 'train':
    print(SEP)
    print('  POLICY TRAINING (Cosmos Policy method)')
    print(SEP)
    print()
    print('  Base model: Cosmos-Predict2-2B-Video2World (32.54 GB VRAM)')
    print()
    print('  COSMOS POLICY NOTES (arXiv 2601.16163, Jan 22 2026):')
    print('    - ALOHA checkpoint NOT compatible with xArm AI hardware')
    print('    - Camera config required: top-down + 2 wrist views (ALOHA)')
    print('    - Our setup: 1 top-down -> must post-train from scratch')
    print('    - Method: latent frame injection + action conditioning')
    print()
    print('  RECOMMENDED: Diffusion Policy (simpler, proven on 1-camera setups)')
    print()
    print('  TRAINING COMMANDS (GPU workstation):')
    print()
    print('  # Option A: Cosmos Policy post-training (if you have 2+ cameras)')
    print('  git clone https://github.com/NVlabs/cosmos-policy')
    print('  pip install -e .')
    print('  python train.py \\')
    print('    --dataset data/cosmos/transfer_clips \\')
    print('    --base_model Cosmos-Predict2-2B-Video2World \\')
    print('    --task xarm_pick_and_place \\')
    print('    --obs_horizon 2 --pred_horizon 16')
    print()
    print('  # Option B: Diffusion Policy (recommended for single camera)')
    print('  pip install diffusers robosuite')
    print('  python train_dp.py \\')
    print('    --data_dir data/cosmos/demos \\')
    print('    --image_keys ["rgb"] \\')
    print('    --action_dim 6 \\')
    print('    --obs_keys ["image", "joint_positions"]')
    print()
    print('  VALIDATION METRICS:')
    print('    - Pick success rate: target > 80%')
    print('    - Place accuracy: target < 1.5cm RMS error')
    print('    - Inference latency: target < 50ms on Pi (ONNX)')
    sys.exit(0)


# ============================================================================
# STEP: EXPORT - ONNX export for Pi deployment
# ============================================================================

if args.step == 'export':
    print(SEP)
    print('  ONNX EXPORT FOR PI DEPLOYMENT')
    print(SEP)
    print()
    print('  ONNX Runtime on Raspberry Pi 5:')
    print('    - CPU inference for lightweight policies')
    print('    - Quantization: INT8 reduces model size 4x and speeds up ~2x')
    print('    - Triton alternative: GPU server + gRPC from Pi (~10ms latency)')
    print()
    print('  EXPORT COMMANDS (on GPU workstation):')
    print()
    print('  # 1. Export from PyTorch to ONNX')
    print('  python export_onnx.py \\')
    print('    --model data/cosmos/model/checkpoint_best.pt \\')
    print('    --output data/cosmos/model/xarm_policy.onnx \\')
    print('    --input_shape "[1,3,224,224]" \\')
    print('    --opset 17')
    print()
    print('  # 2. Verify ONNX export')
    print('  python -c "import onnx; m=onnx.load(\'data/cosmos/model/xarm_policy.onnx\'); onnx.checker.check_model(m); print(\'ONNX OK\')"')
    print()
    print('  # 3. INT8 quantization for Pi')
    print('  from onnxruntime.quantization import quantize_dynamic, QuantType')
    print('  quantize_dynamic(\'xarm_policy.onnx\', \'xarm_policy_int8.onnx\',')
    print('                    weight_type=QuantType.QInt8)')
    print()
    print('  # 4. Copy to Pi')
    print('  scp data/cosmos/model/xarm_policy_int8.onnx pi@192.168.1.163:~/models/')
    print()
    print('  Triton model repository layout:')
    print('    models/')
    print('      xarm_policy/')
    print('        1/')
    print('          model.onnx  (copy here)')
    print('        config.pbtxt')
    sys.exit(0)


# ============================================================================
# STEP: DEPLOY - validate policy on real robot
# ============================================================================

if args.step == 'deploy':
    print(SEP)
    print('  DEPLOY: Validate ONNX Policy on Pi')
    print(SEP)

    model_path = DATA / 'model/xarm_policy_int8.onnx'
    pi_model   = '/home/pi/models/xarm_policy_int8.onnx'

    print()
    print('  DEPLOY SCRIPT (run on Pi):')
    print()
    print('  import onnxruntime as ort')
    print('  import numpy as np')
    print()
    print('  sess = ort.InferenceSession("xarm_policy_int8.onnx")')
    print()
    print('  # Each control step:')
    print('  # 1. Grab frame from Pi camera')
    print('  # 2. Undistort using saved camera intrinsics')
    print('  # 3. Run policy inference')
    print('  # 4. Decode predicted servo positions')
    print('  # 5. Send via /arm/group_move')
    print()

    # Test Pi connectivity
    h = get('/health', t=5)
    if h:
        print(f'  Pi agent: connected  [{h.get("status")}]')
        print(f'  xArm: connected={h.get("xarm")}')
    else:
        print('  Pi agent: not reachable (run on Pi for deployment)')

    print()
    print('  Run validation suite:')
    print('    python vision_pick.py --reps 10 --place left90  # baseline')
    print('    # compare with policy-driven picks:')
    print('    python deploy_policy.py --model xarm_policy_int8.onnx --reps 10')

    sys.exit(0)
