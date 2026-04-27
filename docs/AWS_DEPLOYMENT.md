# AWS Deployment - NIS Protocol v4.0.5

## Quick Deploy (EC2)

```bash
# 1. Launch t3.xlarge instance (Ubuntu 22.04)
# 2. Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 3. Deploy
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git
cd NIS_Protocol
cp .env.example .env
# Edit .env with API keys
./scripts/start-cpu.sh
```

## AWS Services Supported
- EC2, ECS, EKS, Elastic Beanstalk
- Portable paths (no hardcoded /Users/)
- Environment auto-detection

## Security
- Use AWS Secrets Manager for API keys
- Enable CloudWatch logging
- Configure security groups (port 8000)

## Monitoring
- CloudWatch metrics integration
- Health endpoint: /health
- 393 endpoints verified working
