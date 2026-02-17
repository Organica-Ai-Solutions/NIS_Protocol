#!/bin/bash
# H100 Setup Script for NIS Protocol Training
# Run this on the H100 instance after SSH

set -e

echo "🚀 Setting up H100 for NIS Protocol Training"
echo "=============================================="

# 1. Verify GPU access
echo ""
echo "📊 Checking GPU availability..."
nvidia-smi

# 2. Install system dependencies
echo ""
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv htop tmux

# 3. Create working directory
echo ""
echo "📁 Creating workspace..."
mkdir -p ~/organica-ai
cd ~/organica-ai

# 4. Clone repositories
echo ""
echo "📥 Cloning repositories..."

if [ ! -d "NIS_Protocol" ]; then
    git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
    echo "✅ NIS_Protocol cloned"
else
    echo "⏭️  NIS_Protocol already exists"
fi

# Note: NeuroLinux and NIS-HUB are private repos
# You'll need to clone them manually with credentials

# 5. Setup Python environment
echo ""
echo "🐍 Setting up Python environment..."
cd NIS_Protocol
python3 -m venv venv
source venv/bin/activate

# 6. Install PyTorch with CUDA support
echo ""
echo "🔥 Installing PyTorch with CUDA..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7. Install NIS Protocol dependencies
echo ""
echo "📚 Installing NIS Protocol dependencies..."
pip install -r requirements.txt

# 8. Verify PyTorch CUDA
echo ""
echo "✅ Verifying PyTorch CUDA access..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

# 9. Create training directories
echo ""
echo "📂 Creating training directories..."
mkdir -p ~/organica-ai/training/{pinn,kan,vision,voice}
mkdir -p ~/organica-ai/models
mkdir -p ~/organica-ai/logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start a tmux session: tmux new -s training"
echo "2. Activate venv: source ~/organica-ai/NIS_Protocol/venv/bin/activate"
echo "3. Start PINN training: python3 train_pinn.py"
echo ""
echo "GPU Hours Remaining: Check Brev dashboard"
echo "Remember: H100 runs 24/7, keep jobs queued!"
