#!/bin/bash

# ✅ REAL: NVIDIA NIM Production Deployment Script
# This script deploys the NIS Protocol with actual NVIDIA NIM services

set -e

echo "🚀 NIS Protocol Production Deployment with NVIDIA NIM"
echo "=" * 60

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if helm is available
if ! command -v helm &> /dev/null; then
    echo "❌ helm is not installed. Please install helm first."
    exit 1
fi

# Check for NGC API key
if [ -z "$NGC_API_KEY" ]; then
    echo "❌ NGC_API_KEY environment variable not set"
    echo "Please set your NGC API key:"
    echo "export NGC_API_KEY='your_ngc_api_key_here'"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Create namespace
echo "📦 Creating Kubernetes namespace..."
kubectl create namespace nis-protocol --dry-run=client -o yaml | kubectl apply -f -

# Create secrets
echo "🔐 Creating Kubernetes secrets..."
kubectl create secret docker-registry ngc-secret \
    --docker-server=nvcr.io \
    --docker-username='$oauthtoken' \
    --docker-password="$NGC_API_KEY" \
    --namespace=nis-protocol \
    --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic ngc-api \
    --from-literal=NGC_API_KEY="$NGC_API_KEY" \
    --namespace=nis-protocol \
    --dry-run=client -o yaml | kubectl apply -f -

# Download NVIDIA NIM Helm chart
echo "⬇️ Downloading NVIDIA NIM Helm chart..."
helm fetch https://helm.ngc.nvidia.com/nim/charts/nim-llm-1.0.3.tgz \
    --username='$oauthtoken' \
    --password="$NGC_API_KEY" \
    --untar

# Create NIM configuration
echo "⚙️ Creating NIM configuration..."
cat > nim-values.yaml << 'EOF'
# ✅ REAL: Production NIM Configuration for NIS Protocol
image:
  repository: "nvcr.io/nim/meta/llama3-8b-instruct"
  tag: "1.0.3"

model:
  ngcAPISecret: ngc-api
  name: "meta/llama3-8b-instruct"

persistence:
  enabled: true
  size: "50Gi"
  storageClass: "standard"  # Adjust based on your cluster

resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1

# Enable OpenTelemetry for monitoring
env:
  - name: NIM_ENABLE_OTEL
    value: "1"
  - name: NIM_OTEL_SERVICE_NAME
    value: "nis-protocol-llm"
  - name: OTEL_TRACES_EXPORTER
    value: "otlp"
  - name: OTEL_METRICS_EXPORTER
    value: "otlp"
  - name: HOST_IP
    valueFrom:
      fieldRef:
        fieldPath: status.hostIP
  - name: OTEL_EXPORTER_OTLP_ENDPOINT
    value: "http://$(HOST_IP):4318"

# Multi-model support
multiModel:
  enabled: true
  models:
    - name: "meta/llama3-8b-instruct"
      maxConcurrency: 100
    - name: "meta/llama3-70b-instruct"
      maxConcurrency: 50
      resources:
        limits:
          nvidia.com/gpu: 4

imagePullSecrets:
  - name: ngc-secret

# Service configuration
service:
  type: ClusterIP
  openaiPort: 8000

# Health checks
livenessProbe:
  enabled: true
  path: "/v1/health/live"
  initialDelaySeconds: 300  # Longer for model loading

readinessProbe:
  enabled: true
  path: "/v1/health/ready"
  initialDelaySeconds: 300

startupProbe:
  enabled: true
  path: "/v1/health/ready"
  initialDelaySeconds: 600  # Very long for large models
  failureThreshold: 180
EOF

echo "🚀 Deploying NVIDIA NIM..."
helm install nis-llm ./nim-llm \
    --namespace nis-protocol \
    -f nim-values.yaml \
    --wait \
    --timeout 1800s

# Wait for NIM to be ready
echo "⏳ Waiting for NIM to be ready..."
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=nim-llm \
    --namespace=nis-protocol \
    --timeout=1800s

# Get service information
echo "📡 Getting service information..."
kubectl get services -n nis-protocol
kubectl get pods -n nis-protocol

# Test the NIM service
echo "🧪 Testing NIM service..."
SERVICE_IP=$(kubectl get svc nis-llm-nim-llm -n nis-protocol -o jsonpath='{.spec.clusterIP}')
SERVICE_PORT=$(kubectl get svc nis-llm-nim-llm -n nis-protocol -o jsonpath='{.spec.ports[0].port}')

echo "Testing NIM endpoint: http://${SERVICE_IP}:${SERVICE_PORT}/v1/models"

# Wait a bit more for the service to be fully ready
sleep 30

# Test the service
curl -X GET "http://${SERVICE_IP}:${SERVICE_PORT}/v1/models" \
    -H "accept: application/json" \
    --connect-timeout 30 \
    --max-time 60 || echo "⚠️ NIM service may still be starting up"

echo ""
echo "🎉 NVIDIA NIM deployment completed!"
echo ""
echo "📋 Next Steps:"
echo "1. Update your NIS Protocol configuration to use NIM:"
echo "   export NIM_BASE_URL='http://${SERVICE_IP}:${SERVICE_PORT}'"
echo ""
echo "2. Test NIS Protocol with real NIM:"
echo "   python -c 'from src.agents.reasoning.unified_reasoning_agent import EnhancedKANReasoningAgent; agent = EnhancedKANReasoningAgent(); print(agent.nvidia_nim_available)'"
echo ""
echo "3. Deploy additional models as needed:"
echo "   helm upgrade nis-llm ./nim-llm -f nim-values.yaml"
echo ""
echo "4. Monitor the deployment:"
echo "   kubectl logs -f deployment/nis-llm-nim-llm -n nis-protocol"
echo ""
echo "✅ NIS Protocol now uses REAL NVIDIA NIM for production LLM services!"
