# 🚀 NIS Protocol AWS Deployment Guide

## ✅ Production-Ready AWS Migration

This guide covers deploying NIS Protocol to AWS with proper path configuration, environment variables, and production best practices.

---

## 📋 **Prerequisites**

### **AWS Account Setup**
- ✅ AWS account with admin access
- ✅ AWS CLI installed and configured
- ✅ Domain name (for API endpoint)

### **NIS Protocol Requirements**
- ✅ Docker images built and tested
- ✅ Environment variables configured
- ✅ LLM API keys (OpenAI, Anthropic, etc.)
- ✅ SSL certificates (for HTTPS)

---

## 🏗️ **Deployment Architecture Options**

### **Option 1: EC2 with Docker Compose** (Recommended for MVP)
```
Internet → ALB → EC2 (Docker Compose) → Redis/RDS
         ↓
      Route53
```

**Pros:** Simple, full control, fast to deploy
**Cons:** Manual scaling, requires monitoring setup

### **Option 2: ECS Fargate** (Recommended for Production)
```
Internet → ALB → ECS Fargate → ElastiCache/RDS
         ↓          ↓
      Route53   Auto Scaling
```

**Pros:** Serverless, auto-scaling, managed
**Cons:** More complex setup, slightly higher cost

### **Option 3: EKS (Kubernetes)** (For Large Scale)
```
Internet → ALB → EKS Cluster → ElastiCache/RDS
         ↓          ↓
      Route53   HPA/Cluster Auto Scaling
```

**Pros:** Full orchestration, multi-region support
**Cons:** Most complex, highest initial cost

---

## 🚀 **Quick Deploy: EC2 + Docker Compose**

### **Step 1: Launch EC2 Instance**

```bash
# Launch Ubuntu 22.04 EC2 instance
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-your-security-group \
  --subnet-id subnet-your-subnet \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=nis-protocol-backend}]'
```

**Instance Sizing:**
- **Development**: `t3.medium` (2 vCPU, 4GB RAM)
- **Production**: `t3.xlarge` (4 vCPU, 16GB RAM)
- **High Load**: `c5.2xlarge` (8 vCPU, 16GB RAM)

### **Step 2: SSH and Setup**

```bash
# SSH to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.24.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create project directory
sudo mkdir -p /opt/nis-protocol
sudo chown ubuntu:ubuntu /opt/nis-protocol
```

### **Step 3: Deploy NIS Protocol**

```bash
# Clone or copy project
cd /opt/nis-protocol
git clone https://github.com/Organica-Ai-Solutions/NIS_Protocol.git .

# Or use rsync from local:
# rsync -avz --exclude 'node_modules' --exclude '__pycache__' \
#   /Users/diegofuego/Desktop/NIS_Protocol/ \
#   ubuntu@your-ec2-ip:/opt/nis-protocol/

# Copy AWS environment config
cp configs/deployment.aws.env .env

# Edit environment variables
nano .env
# Set:
# - OPENAI_API_KEY
# - ANTHROPIC_API_KEY  
# - NIS_BACKEND_URL=https://api.yourdomain.com
# - NIS_MCP_API_KEY=your-secure-key

# Set environment variable for all sessions
echo 'export NIS_PROJECT_ROOT=/opt/nis-protocol' | sudo tee -a /etc/environment
echo 'export PYTHONPATH=/opt/nis-protocol' | sudo tee -a /etc/environment
source /etc/environment

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f backend
```

### **Step 4: Configure ALB (Application Load Balancer)**

```bash
# Create target group
aws elbv2 create-target-group \
  --name nis-protocol-tg \
  --protocol HTTP \
  --port 80 \
  --vpc-id vpc-your-vpc \
  --health-check-path /health

# Register EC2 instance
aws elbv2 register-targets \
  --target-group-arn arn:aws:elasticloadbalancing:... \
  --targets Id=i-your-instance-id

# Create ALB with SSL
aws elbv2 create-load-balancer \
  --name nis-protocol-alb \
  --subnets subnet-1 subnet-2 \
  --security-groups sg-your-alb-sg

# Add HTTPS listener with ACM certificate
aws elbv2 create-listener \
  --load-balancer-arn arn:aws:elasticloadbalancing:... \
  --protocol HTTPS \
  --port 443 \
  --certificates CertificateArn=arn:aws:acm:... \
  --default-actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
```

### **Step 5: Configure DNS (Route53)**

```bash
# Get ALB DNS name
ALB_DNS=$(aws elbv2 describe-load-balancers \
  --names nis-protocol-alb \
  --query 'LoadBalancers[0].DNSName' \
  --output text)

# Create A record
aws route53 change-resource-record-sets \
  --hosted-zone-id Z1234567890ABC \
  --change-batch '{
    "Changes": [{
      "Action": "CREATE",
      "ResourceRecordSet": {
        "Name": "api.yourdomain.com",
        "Type": "A",
        "AliasTarget": {
          "HostedZoneId": "Z35SXDOTRQ7X7K",
          "DNSName": "'$ALB_DNS'",
          "EvaluateTargetHealth": true
        }
      }
    }]
  }'
```

---

## 🔒 **Security Configuration**

### **1. Security Groups**

```bash
# ALB Security Group (allow HTTPS from internet)
aws ec2 authorize-security-group-ingress \
  --group-id sg-alb \
  --protocol tcp \
  --port 443 \
  --cidr 0.0.0.0/0

# EC2 Security Group (allow HTTP from ALB only)
aws ec2 authorize-security-group-ingress \
  --group-id sg-ec2 \
  --protocol tcp \
  --port 80 \
  --source-group sg-alb

# Redis Security Group (allow from EC2 only)
aws ec2 authorize-security-group-ingress \
  --group-id sg-redis \
  --protocol tcp \
  --port 6379 \
  --source-group sg-ec2
```

### **2. IAM Roles**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::nis-protocol-artifacts/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "dynamodb:PutItem",
        "dynamodb:GetItem",
        "dynamodb:Query",
        "dynamodb:UpdateItem"
      ],
      "Resource": "arn:aws:dynamodb:*:*:table/nis-jobs"
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/nis-protocol:*"
    }
  ]
}
```

### **3. Secrets Manager**

```bash
# Store API keys in Secrets Manager
aws secretsmanager create-secret \
  --name nis-protocol/api-keys \
  --secret-string '{
    "OPENAI_API_KEY": "sk-your-key",
    "ANTHROPIC_API_KEY": "sk-ant-your-key",
    "NIS_MCP_API_KEY": "your-mcp-key"
  }'

# Update .env to fetch from Secrets Manager
# (Add script to fetch secrets on container start)
```

---

## 📊 **Monitoring & Observability**

### **1. CloudWatch Logs**

```bash
# Configure Docker logging to CloudWatch
cat > /etc/docker/daemon.json <<EOF
{
  "log-driver": "awslogs",
  "log-opts": {
    "awslogs-region": "us-east-1",
    "awslogs-group": "/aws/nis-protocol",
    "awslogs-create-group": "true"
  }
}
EOF

sudo systemctl restart docker
```

### **2. CloudWatch Metrics**

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i -E ./amazon-cloudwatch-agent.deb

# Configure metrics
cat > /opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-config.json <<EOF
{
  "metrics": {
    "namespace": "NISProtocol",
    "metrics_collected": {
      "cpu": {
        "measurement": [{"name": "cpu_usage_idle"}]
      },
      "memory": {
        "measurement": [{"name": "mem_used_percent"}]
      },
      "disk": {
        "measurement": [{"name": "used_percent"}]
      }
    }
  }
}
EOF

sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
  -a fetch-config \
  -m ec2 \
  -s \
  -c file:/opt/aws/amazon-cloudwatch-agent/etc/cloudwatch-config.json
```

### **3. Alarms**

```bash
# High CPU alarm
aws cloudwatch put-metric-alarm \
  --alarm-name nis-protocol-high-cpu \
  --alarm-description "CPU utilization > 80%" \
  --metric-name CPUUtilization \
  --namespace AWS/EC2 \
  --statistic Average \
  --period 300 \
  --threshold 80 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:123456789:nis-alerts

# Error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name nis-protocol-high-errors \
  --alarm-description "Error rate > 5%" \
  --metric-name 5XXError \
  --namespace AWS/ApplicationELB \
  --statistic Average \
  --period 60 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2 \
  --alarm-actions arn:aws:sns:us-east-1:123456789:nis-alerts
```

---

## 🔄 **Auto-Scaling (Optional)**

```bash
# Create launch template
aws ec2 create-launch-template \
  --launch-template-name nis-protocol-template \
  --version-description "v1" \
  --launch-template-data '{
    "ImageId": "ami-0c55b159cbfafe1f0",
    "InstanceType": "t3.xlarge",
    "KeyName": "your-key",
    "SecurityGroupIds": ["sg-your-sg"],
    "UserData": "<base64-encoded-startup-script>",
    "TagSpecifications": [{
      "ResourceType": "instance",
      "Tags": [{"Key": "Name", "Value": "nis-protocol-auto"}]
    }]
  }'

# Create Auto Scaling Group
aws autoscaling create-auto-scaling-group \
  --auto-scaling-group-name nis-protocol-asg \
  --launch-template LaunchTemplateName=nis-protocol-template \
  --min-size 2 \
  --max-size 10 \
  --desired-capacity 2 \
  --target-group-arns arn:aws:elasticloadbalancing:... \
  --vpc-zone-identifier "subnet-1,subnet-2"

# Scaling policies
aws autoscaling put-scaling-policy \
  --auto-scaling-group-name nis-protocol-asg \
  --policy-name scale-up \
  --scaling-adjustment 2 \
  --adjustment-type ChangeInCapacity \
  --cooldown 300
```

---

## 🧪 **Testing Production Deployment**

```bash
# Health check
curl https://api.yourdomain.com/health

# MCP capabilities
curl -X POST https://api.yourdomain.com/research/capabilities

# Robotics control
curl -X POST https://api.yourdomain.com/robotics/forward_kinematics \
  -H "Content-Type: application/json" \
  -d '{
    "robot_id": "prod_drone_001",
    "robot_type": "drone",
    "joint_angles": [0, 0.785, 1.57, 0, 0.785, 0]
  }'

# Cost tracking
curl https://api.yourdomain.com/metrics/cost_report
```

---

## 💰 **Cost Optimization**

### **Monthly Cost Estimate**

| Service | Config | Monthly Cost |
|---------|--------|--------------|
| EC2 (t3.xlarge) | 2 instances | ~$220 |
| ALB | Standard | ~$23 |
| ElastiCache (Redis) | cache.t3.medium | ~$50 |
| RDS (PostgreSQL) | db.t3.medium | ~$70 |
| S3 | 100GB storage | ~$3 |
| CloudWatch | Logs + Metrics | ~$15 |
| **Total** | | **~$381/month** |

### **Cost Saving Tips**
- ✅ Use Reserved Instances (save 40-60%)
- ✅ Use Spot Instances for workers (save 70-90%)
- ✅ Enable S3 Intelligent-Tiering
- ✅ Set CloudWatch log retention to 7 days
- ✅ Use ALB Request Routing to reduce instance count

---

## 📚 **Next Steps**

1. ✅ Deploy to AWS EC2
2. ✅ Configure ALB + SSL
3. ✅ Set up monitoring
4. ✅ Test all endpoints
5. ✅ Configure ChatGPT/Claude with production URL
6. ✅ Set up CI/CD (GitHub Actions → AWS)
7. ✅ Plan multi-region deployment

---

## 🆘 **Support & Troubleshooting**

**Logs:**
```bash
# Backend logs
docker logs nis-backend -f

# System logs
sudo journalctl -u docker -f

# CloudWatch logs
aws logs tail /aws/nis-protocol --follow
```

**Common Issues:**
- ❌ **502 Bad Gateway** → Check backend health (`docker ps`)
- ❌ **Connection timeout** → Verify security groups
- ❌ **High latency** → Scale up EC2 instance or add more instances
- ❌ **API key errors** → Check Secrets Manager permissions

---

**AWS MAP Program Status:** Applied (pending approval)
**NVIDIA Inception:** Accepted (no DGX Cloud access yet)

**Ready for production deployment! 🚀**

