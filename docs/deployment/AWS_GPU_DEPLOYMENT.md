# AWS ECS GPU Deployment Guide for Cloudelligent Team

## Issue Fixed: `/opt/nvidia/nvidia_entrypoint.sh: line 67: exec: python: not found`

**Root Cause:** NVIDIA CUDA base images expect `python` command, but we only installed `python3.11`.

**Fix Applied:** Added symlinks in Dockerfile:
```bash
ln -s /usr/bin/python3.11 /usr/bin/python
ln -s /usr/bin/python3.11 /usr/bin/python3
```

---

## ECS Task Definition Configuration for GPU (g4dn.xlarge with Tesla T4)

### 1. Task Definition JSON - GPU Resource Requirements

```json
{
  "family": "nis-protocol-backend-gpu",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-ecr-repo/nis-protocol:gpu",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "environment": [
        {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
        {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
        {"name": "CUDA_HOME", "value": "/usr/local/cuda"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/nis-protocol-gpu",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "backend"
        }
      }
    }
  ]
}
```

### 2. ECS Cluster Configuration

**Instance Type:** `g4dn.xlarge` (Tesla T4, 4 vCPUs, 16 GB RAM)

**AMI:** Use **ECS GPU-Optimized AMI**
```bash
# Latest ECS GPU-Optimized AMI (includes NVIDIA drivers + Container Toolkit)
aws ssm get-parameter \
  --name /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended \
  --region us-east-1 \
  --query "Parameter.Value" \
  --output text
```

### 3. EC2 Instance User Data (Auto-registers to ECS cluster)

```bash
#!/bin/bash
echo ECS_CLUSTER=nis-protocol-gpu >> /etc/ecs/ecs.config
echo ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config
```

### 4. Docker Run Command (for testing on EC2 directly)

```bash
# Test the container on the EC2 instance
docker run --rm --gpus all \
  -p 8000:8000 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  your-ecr-repo/nis-protocol:gpu
```

---

## Verification Steps

### On EC2 Instance (g4dn.xlarge)

1. **Check NVIDIA drivers:**
   ```bash
   nvidia-smi
   ```
   Expected output: Tesla T4 GPU info, CUDA 12.1+

2. **Check NVIDIA Container Toolkit:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```
   Should show GPU info inside container.

3. **Check ECS agent GPU support:**
   ```bash
   cat /etc/ecs/ecs.config | grep GPU
   ```
   Should show: `ECS_ENABLE_GPU_SUPPORT=true`

### Inside Running Container

```bash
# Exec into container
docker exec -it <container_id> bash

# Verify Python
python --version  # Should show: Python 3.11.x

# Verify CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
# Expected: CUDA Available: True, Version: 12.1
```

---

## Troubleshooting

### Issue: "NVIDIA Driver was not detected"
**Cause:** Container not launched with `--gpus all` flag or ECS task missing GPU resource requirement.

**Fix for ECS:** Ensure Task Definition includes:
```json
"resourceRequirements": [{"type": "GPU", "value": "1"}]
```

**Fix for Docker:** Add `--gpus all` flag.

### Issue: "exec: python: not found"
**Cause:** NVIDIA base image entrypoint expects `python` command.

**Fix:** Already applied in Dockerfile (symlinks added).

### Issue: Container crashes on startup
**Check logs:**
```bash
# ECS
aws logs tail /ecs/nis-protocol-gpu --follow

# Docker
docker logs <container_id>
```

---

## Build and Push to ECR

```bash
# Build GPU image
docker build -t nis-protocol:gpu -f Dockerfile .

# Tag for ECR
docker tag nis-protocol:gpu <account-id>.dkr.ecr.us-east-1.amazonaws.com/nis-protocol:gpu

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/nis-protocol:gpu
```

---

## Key Points for Mandeep

1. ✅ **No manual driver installation needed** - ECS GPU-Optimized AMI includes everything
2. ✅ **Python symlink issue fixed** - Dockerfile now creates `python` -> `python3.11` symlink
3. ✅ **Task Definition must specify GPU resource** - See JSON example above
4. ✅ **Use ECS GPU-Optimized AMI** - Not regular ECS-Optimized AMI
5. ✅ **Set `ECS_ENABLE_GPU_SUPPORT=true`** in instance user data

---

## Contact

If issues persist after rebuilding with the fixed Dockerfile, check:
1. ECS Task Definition has `resourceRequirements` for GPU
2. EC2 instance is using ECS GPU-Optimized AMI
3. `nvidia-smi` works on the host EC2 instance
