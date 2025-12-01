#!/bin/bash
# ========================================================================
# NVIDIA DGX Cloud Lepton Deployment Script for NIS Protocol
# Deploy to multi-cloud GPU marketplace
# ========================================================================

set -e

# Configuration
PROJECT_NAME="nis-protocol"
IMAGE_TAG="${IMAGE_TAG:-v4.0}"
LEPTON_WORKSPACE="${LEPTON_WORKSPACE:-nis-protocol-workspace}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  DGX Cloud Lepton - NIS Protocol${NC}"
echo -e "${CYAN}========================================${NC}"

# Check prerequisites
echo -e "\n${YELLOW}[1/5] Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check for Lepton CLI (when available)
if command -v lepton &> /dev/null; then
    echo -e "${GREEN}✓ Lepton CLI found${NC}"
    LEPTON_CLI=true
else
    echo -e "${YELLOW}⚠ Lepton CLI not installed (will use API)${NC}"
    LEPTON_CLI=false
fi

if [ -z "$NVIDIA_API_KEY" ]; then
    echo -e "${YELLOW}⚠ NVIDIA_API_KEY not set - some features may be limited${NC}"
fi

echo -e "${GREEN}✓ Prerequisites check complete${NC}"

# Build Docker image
echo -e "\n${YELLOW}[2/5] Building Docker image...${NC}"
docker build -f deploy/nvidia/Dockerfile.nvcf -t ${PROJECT_NAME}:${IMAGE_TAG} .
echo -e "${GREEN}✓ Image built successfully${NC}"

# Create Lepton deployment configuration
echo -e "\n${YELLOW}[3/5] Creating Lepton deployment config...${NC}"

cat << EOF > deploy/nvidia/lepton-config.yaml
# DGX Cloud Lepton Deployment Configuration
# NIS Protocol v4.0

apiVersion: lepton/v1
kind: Deployment
metadata:
  name: ${PROJECT_NAME}
  workspace: ${LEPTON_WORKSPACE}
  labels:
    app: nis-protocol
    version: "${IMAGE_TAG}"
    
spec:
  # Container configuration
  container:
    image: ${PROJECT_NAME}:${IMAGE_TAG}
    port: 8000
    
  # Resource requirements
  resources:
    # GPU options: T4, L4, L40S, A10G, A100, H100
    gpu: L40S
    gpuCount: 1
    memory: 32Gi
    cpu: 8
    
  # Scaling configuration
  scaling:
    minReplicas: 1
    maxReplicas: 10
    targetGPUUtilization: 80
    
  # Health checks
  health:
    path: /health
    initialDelaySeconds: 60
    periodSeconds: 30
    
  # Environment variables
  env:
    - name: NIS_ENV
      value: production
    - name: LOG_LEVEL
      value: INFO
    - name: NVIDIA_API_KEY
      valueFrom:
        secretRef:
          name: nvidia-api-key
          
  # Regions (select based on availability)
  regions:
    preferred:
      - us-west-2    # CoreWeave
      - us-east-1    # Lambda
      - eu-west-1    # Nebius
    fallback:
      - any
      
  # Cost optimization
  costOptimization:
    spotInstances: true
    maxHourlyBudget: 10.00
EOF

echo -e "${GREEN}✓ Lepton config created: deploy/nvidia/lepton-config.yaml${NC}"

# Create secrets template
echo -e "\n${YELLOW}[4/5] Creating secrets template...${NC}"

cat << EOF > deploy/nvidia/lepton-secrets.yaml
# DGX Cloud Lepton Secrets Configuration
# IMPORTANT: Do not commit this file with real values!

apiVersion: lepton/v1
kind: Secret
metadata:
  name: nis-protocol-secrets
  workspace: ${LEPTON_WORKSPACE}
  
data:
  # NVIDIA API Key for NIM access
  nvidia-api-key: <base64-encoded-nvidia-api-key>
  
  # OpenAI API Key (optional)
  openai-api-key: <base64-encoded-openai-key>
  
  # Anthropic API Key (optional)
  anthropic-api-key: <base64-encoded-anthropic-key>
EOF

echo -e "${GREEN}✓ Secrets template created${NC}"

# Summary
echo -e "\n${YELLOW}[5/5] Deployment summary...${NC}"

echo -e "\n${CYAN}========================================${NC}"
echo -e "${GREEN}DGX Cloud Lepton package ready!${NC}"
echo -e "${CYAN}========================================${NC}"

echo -e "\n${BLUE}Files created:${NC}"
echo -e "  • deploy/nvidia/Dockerfile.nvcf"
echo -e "  • deploy/nvidia/lepton-config.yaml"
echo -e "  • deploy/nvidia/lepton-secrets.yaml"

echo -e "\n${YELLOW}Deployment steps (after Lepton access approved):${NC}"
echo -e "  1. Login to DGX Cloud Lepton portal"
echo -e "  2. Create workspace: ${LEPTON_WORKSPACE}"
echo -e "  3. Push image to Lepton registry"
echo -e "  4. Apply config: lepton deploy -f deploy/nvidia/lepton-config.yaml"
echo -e "  5. Monitor: lepton status ${PROJECT_NAME}"

echo -e "\n${YELLOW}GPU options available on Lepton marketplace:${NC}"
echo -e "  • ${GREEN}T4${NC}     - Budget inference (~\$0.50/hr)"
echo -e "  • ${GREEN}L4${NC}     - Balanced (~\$0.80/hr)"
echo -e "  • ${GREEN}L40S${NC}   - High performance (~\$1.50/hr)"
echo -e "  • ${GREEN}A10G${NC}   - Production (~\$1.20/hr)"
echo -e "  • ${GREEN}A100${NC}   - Enterprise (~\$3.00/hr)"
echo -e "  • ${GREEN}H100${NC}   - Maximum performance (~\$4.50/hr)"

echo -e "\n${CYAN}Estimated monthly cost (1 GPU, 24/7):${NC}"
echo -e "  L40S: ~\$1,080/month"
echo -e "  With auto-scaling: ~\$300-500/month (typical usage)"
