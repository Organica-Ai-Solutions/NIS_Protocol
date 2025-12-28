# ECS Task Definitions

This directory contains the ECS task definitions for deploying NIS Protocol to AWS.

## Files

- `ecs-task-definition-backend.json` - Backend service (GPU-enabled on g4dn.xlarge)
- `ecs-task-definition-runner.json` - Runner service (Fargate)

## Usage

### Register Task Definitions

```bash
# Register backend task definition
aws ecs register-task-definition \
  --cli-input-json file://deploy/ecs-task-definition-backend.json \
  --region us-east-2

# Register runner task definition
aws ecs register-task-definition \
  --cli-input-json file://deploy/ecs-task-definition-runner.json \
  --region us-east-2
```

### Create Services (First Time Only)

```bash
# Create backend service
aws ecs create-service \
  --cluster nis-ecs-cluster \
  --service-name nis-backend-service \
  --task-definition nis-backend-task \
  --desired-count 1 \
  --launch-type EC2 \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-east-2:774518279463:targetgroup/xxx,containerName=backend,containerPort=8000" \
  --region us-east-2

# Create runner service
aws ecs create-service \
  --cluster nis-ecs-cluster \
  --service-name nis-runner-service \
  --task-definition nis-runner-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=DISABLED}" \
  --region us-east-2
```

### Update Services (After Changes)

```bash
# Update backend service
aws ecs update-service \
  --cluster nis-ecs-cluster \
  --service nis-backend-service \
  --task-definition nis-backend-task \
  --force-new-deployment \
  --region us-east-2

# Update runner service
aws ecs update-service \
  --cluster nis-ecs-cluster \
  --service nis-runner-service \
  --task-definition nis-runner-task \
  --force-new-deployment \
  --region us-east-2
```

## Task Definition Details

### Backend
- **Family**: nis-backend-task
- **Network Mode**: awsvpc
- **Requires**: EC2 (for GPU)
- **CPU**: 1024 (1 vCPU)
- **Memory**: 3072 MB
- **GPU**: 1 NVIDIA T4
- **Port**: 8000
- **Secrets**: 4 API keys from AWS Secrets Manager
- **Logs**: /ecs/backend

### Runner
- **Family**: nis-runner-task
- **Network Mode**: awsvpc
- **Requires**: Fargate
- **CPU**: 1024 (1 vCPU)
- **Memory**: 3072 MB
- **Port**: 8001
- **Logs**: /ecs/runner

## CloudWatch Log Groups

Ensure these log groups exist before deploying:

```bash
aws logs create-log-group --log-group-name /ecs/backend --region us-east-2
aws logs create-log-group --log-group-name /ecs/runner --region us-east-2
```

## Secrets Manager

Verify secrets exist:

```bash
aws secretsmanager list-secrets --region us-east-2 | grep nis/
```

Expected secrets:
- nis/openai-api-key-x0UEEi
- nis/anthropic-api-key-00TnSn
- nis/google-api-key-UpwtiO

## Deployment via GitHub Actions

The `.github/workflows/deploy-aws.yml` workflow automatically:
1. Builds Docker images
2. Pushes to ECR
3. Updates task definitions with new image tags
4. Deploys to ECS services

Just push to `main` branch to trigger deployment.
