#!/bin/bash
# ========================================================================
# NVIDIA Cloud Functions (NVCF) Deployment Script for NIS Protocol
# Deploy to DGX Cloud Serverless Inference
# ========================================================================

set -e

# Configuration
NGC_ORG="${NGC_ORG:-organica-ai}"
NGC_TEAM="${NGC_TEAM:-nis-protocol}"
IMAGE_NAME="nis-protocol"
IMAGE_TAG="${IMAGE_TAG:-v4.0}"
NVCF_FUNCTION_NAME="nis-protocol-inference"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  NVIDIA NVCF Deployment - NIS Protocol${NC}"
echo -e "${BLUE}========================================${NC}"

# Check prerequisites
echo -e "\n${YELLOW}[1/6] Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

if ! command -v ngc &> /dev/null; then
    echo -e "${RED}Error: NGC CLI is not installed${NC}"
    echo -e "${YELLOW}Install from: https://ngc.nvidia.com/setup/installers/cli${NC}"
    exit 1
fi

if [ -z "$NGC_API_KEY" ]; then
    echo -e "${RED}Error: NGC_API_KEY environment variable not set${NC}"
    echo -e "${YELLOW}Get your key from: https://ngc.nvidia.com/setup/api-key${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All prerequisites met${NC}"

# Login to NGC
echo -e "\n${YELLOW}[2/6] Logging into NGC registry...${NC}"
echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
echo -e "${GREEN}✓ Logged into NGC${NC}"

# Build Docker image
echo -e "\n${YELLOW}[3/6] Building Docker image for NVCF...${NC}"
docker build -f deploy/nvidia/Dockerfile.nvcf -t ${IMAGE_NAME}:${IMAGE_TAG} .
echo -e "${GREEN}✓ Image built successfully${NC}"

# Tag for NGC
echo -e "\n${YELLOW}[4/6] Tagging image for NGC...${NC}"
NGC_IMAGE="nvcr.io/${NGC_ORG}/${NGC_TEAM}/${IMAGE_NAME}:${IMAGE_TAG}"
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${NGC_IMAGE}
echo -e "${GREEN}✓ Tagged as ${NGC_IMAGE}${NC}"

# Push to NGC
echo -e "\n${YELLOW}[5/6] Pushing to NGC registry...${NC}"
docker push ${NGC_IMAGE}
echo -e "${GREEN}✓ Pushed to NGC${NC}"

# Create NVCF Function
echo -e "\n${YELLOW}[6/6] Creating NVCF Function...${NC}"
cat << EOF > /tmp/nvcf-function.json
{
  "name": "${NVCF_FUNCTION_NAME}",
  "inferenceUrl": "/chat",
  "healthUri": "/health",
  "containerImage": "${NGC_IMAGE}",
  "containerArgs": "",
  "containerEnvironment": [
    {"key": "NIS_ENV", "value": "production"},
    {"key": "LOG_LEVEL", "value": "INFO"},
    {"key": "NVIDIA_API_KEY", "value": "\${secrets.NVIDIA_API_KEY}"}
  ],
  "models": [],
  "resources": {
    "gpu": "L40S",
    "gpuCount": 1,
    "memoryInGb": 32
  },
  "apiBodyFormat": "CUSTOM",
  "description": "NIS Protocol - Neural Intelligence System with Physics-Informed AI"
}
EOF

echo -e "${GREEN}✓ NVCF function configuration created${NC}"
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Deployment package ready!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${YELLOW}Next steps:${NC}"
echo -e "1. Wait for NVCF access approval"
echo -e "2. Run: ngc cloud-function function create -f /tmp/nvcf-function.json"
echo -e "3. Deploy: ngc cloud-function function deploy ${NVCF_FUNCTION_NAME}"
echo -e "\n${YELLOW}Image: ${NGC_IMAGE}${NC}"
