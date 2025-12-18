# Message for Mandeep Singh & Momina Qureshi (Cloudelligent Team)

Hi Mandeep & Momina,

I've identified and fixed the GPU Docker compatibility issues you reported. The container should now work properly on your g4dn.xlarge instances with Tesla T4 GPUs.

---

## 🔧 Issues Fixed

### 1. **Python Executable Not Found**
**Error:** `/opt/nvidia/nvidia_entrypoint.sh: line 67: exec: python: not found`

**Root Cause:** NVIDIA CUDA base images expect the `python` command, but we only installed `python3.11` without creating the necessary symlinks.

**Fix Applied:** Updated `Dockerfile` to create symlinks in both builder and runtime stages:
```bash
ln -s /usr/bin/python3.11 /usr/bin/python
ln -s /usr/bin/python3.11 /usr/bin/python3
```

### 2. **NVIDIA Driver Warning**
**Warning:** `The NVIDIA Driver was not detected. GPU functionality will not be available.`

**Root Cause:** This happens when the container isn't launched with proper GPU access configuration.

**Fix Required:** See ECS Task Definition configuration below.

---

## ✅ Action Items for Deployment

### Step 1: Pull Latest Code & Rebuild
```bash
git pull origin main
docker build -t nis-protocol:gpu -f Dockerfile .
```

### Step 2: Tag and Push to ECR
```bash
# Replace <account-id> and <region> with your values
docker tag nis-protocol:gpu <account-id>.dkr.ecr.<region>.amazonaws.com/nis-protocol:gpu

aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com

docker push <account-id>.dkr.ecr.<region>.amazonaws.com/nis-protocol:gpu
```

### Step 3: Update ECS Task Definition

**Critical:** Your Task Definition MUST include GPU resource requirements. Here's the complete configuration:

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
      "image": "<account-id>.dkr.ecr.<region>.amazonaws.com/nis-protocol:gpu",
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
        {"name": "CUDA_HOME", "value": "/usr/local/cuda"},
        {"name": "NIS_ENV", "value": "production"},
        {"name": "NVIDIA_NIM_ENABLED", "value": "true"}
      ],
      "secrets": [
        {"name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "REDIS_HOST", "valueFrom": "arn:aws:secretsmanager:..."},
        {"name": "KAFKA_BOOTSTRAP_SERVERS", "valueFrom": "arn:aws:secretsmanager:..."}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/nis-protocol-gpu",
          "awslogs-region": "<region>",
          "awslogs-stream-prefix": "backend"
        }
      }
    }
  ]
}
```

**Key Points:**
- ✅ `resourceRequirements` with `"type": "GPU"` is **REQUIRED**
- ✅ `NVIDIA_VISIBLE_DEVICES=all` enables GPU access
- ✅ Set `NVIDIA_NIM_ENABLED=true` for Tesla GPU optimization

### Step 4: Verify EC2 Instance Configuration

**Instance Type:** `g4dn.xlarge` (Tesla T4, 4 vCPUs, 16 GB RAM)

**AMI:** Must use **ECS GPU-Optimized AMI** (not regular ECS AMI)

To get the latest GPU-optimized AMI ID:
```bash
aws ssm get-parameter \
  --name /aws/service/ecs/optimized-ami/amazon-linux-2/gpu/recommended \
  --region <region> \
  --query "Parameter.Value" \
  --output text | jq -r '.image_id'
```

**User Data Script** (for EC2 Launch Template):
```bash
#!/bin/bash
echo ECS_CLUSTER=nis-protocol-gpu >> /etc/ecs/ecs.config
echo ECS_ENABLE_GPU_SUPPORT=true >> /etc/ecs/ecs.config
```

---

## 🧪 Verification Steps

### On EC2 Instance (SSH into g4dn.xlarge)

1. **Check NVIDIA drivers are installed:**
   ```bash
   nvidia-smi
   ```
   Expected: Should show Tesla T4 GPU info, CUDA 12.1+

2. **Verify ECS GPU support is enabled:**
   ```bash
   cat /etc/ecs/ecs.config | grep GPU
   ```
   Expected: `ECS_ENABLE_GPU_SUPPORT=true`

3. **Test GPU access in Docker:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```
   Expected: Should show GPU info inside container

### After ECS Task is Running

1. **Check container logs:**
   ```bash
   aws logs tail /ecs/nis-protocol-gpu --follow
   ```
   Look for: No Python errors, successful startup

2. **Test API endpoint:**
   ```bash
   curl http://<alb-endpoint>/health
   ```
   Expected: `{"status": "healthy"}`

3. **Verify GPU is detected inside container:**
   ```bash
   # Get container ID from ECS
   docker exec -it <container-id> python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
   ```
   Expected: `CUDA: True, Version: 12.1`

---

## 📋 Environment Variables Checklist

Refer to `.env.ecs-minimal` in the repo for the complete list. **Required variables:**

- ✅ At least ONE LLM API key (OpenAI/Anthropic/DeepSeek/Google)
- ✅ `REDIS_HOST` (ElastiCache endpoint)
- ✅ `KAFKA_BOOTSTRAP_SERVERS` (MSK bootstrap servers, comma-separated)
- ✅ `JWT_SECRET_KEY` (generate a secure random string)
- ✅ `NVIDIA_NIM_ENABLED=true` (for GPU instances)
- ✅ `NGC_API_KEY` (if using NVIDIA NIM features)

---

## 🚨 Common Issues & Solutions

### Issue: "exec: python: not found"
**Status:** ✅ **FIXED** in latest Dockerfile (commit `fa7ca28`)

### Issue: "NVIDIA Driver was not detected"
**Solution:** Ensure Task Definition has `resourceRequirements` with GPU type (see Step 3 above)

### Issue: Container crashes on startup
**Check:**
1. CloudWatch Logs: `/ecs/nis-protocol-gpu`
2. Verify all required environment variables are set
3. Ensure Redis and Kafka endpoints are accessible from ECS tasks

### Issue: GPU not accessible in container
**Check:**
1. EC2 instance is using ECS GPU-Optimized AMI
2. `ECS_ENABLE_GPU_SUPPORT=true` in `/etc/ecs/ecs.config`
3. Task Definition has GPU resource requirement
4. Run `nvidia-smi` on host to verify drivers work

---

## 📚 Additional Resources

I've created a comprehensive deployment guide in the repo:
- **File:** `AWS_GPU_DEPLOYMENT.md`
- **Location:** Root of the repository
- **Contents:** Detailed ECS configuration, troubleshooting, and verification steps

---

## 🎯 Summary

**What Changed:**
1. Fixed Python symlink issue in Dockerfile (both stages)
2. Added comprehensive AWS GPU deployment documentation
3. Provided complete ECS Task Definition template

**What You Need to Do:**
1. Pull latest code from `main` branch
2. Rebuild Docker image
3. Update ECS Task Definition with GPU resource requirements
4. Ensure EC2 instances use ECS GPU-Optimized AMI
5. Set `ECS_ENABLE_GPU_SUPPORT=true` in instance user data

**Expected Result:**
Container will start successfully on g4dn.xlarge with Tesla T4 GPU fully accessible.

---

Let me know if you encounter any issues after rebuilding. The fixes are tested and should resolve the errors you reported.

Best regards,
Diego Torres
Organica AI Solutions
