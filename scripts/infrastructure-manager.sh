#!/bin/bash
# NIS Protocol Infrastructure Manager
# Manage AWS infrastructure on-demand to save costs

set -e

export AWS_DEFAULT_REGION="us-east-2"

if [ -z "${AWS_PROFILE:-}" ]; then
    if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
        echo "ERROR: AWS credentials not set. Export AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or set AWS_PROFILE." 1>&2
        exit 1
    fi
fi

INFRA_DIR="/Users/diegofuego/Desktop/NIS_Protocol/infrastructure"
TERRAGRUNT_DIR="$INFRA_DIR/terragrunt-live/prod/us-east-2"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

function show_status() {
    echo -e "${BLUE}=== NIS Protocol Infrastructure Status ===${NC}"
    echo ""
    
    # Check ECS services
    echo "ECS Services:"
    aws ecs describe-services --cluster nis-ecs-cluster --services backend-service runner-service \
        --query 'services[*].[serviceName,status,desiredCount,runningCount]' --output table 2>/dev/null || echo "  Not running"
    
    # Check Kafka
    echo -e "\nKafka Cluster:"
    aws kafka list-clusters --query 'ClusterInfoList[*].[ClusterName,State,NumberOfBrokerNodes]' --output table 2>/dev/null || echo "  Not running"
    
    # Check Redis
    echo -e "\nRedis Cluster:"
    aws elasticache describe-cache-clusters --query 'CacheClusters[*].[CacheClusterId,CacheClusterStatus,CacheNodeType]' --output table 2>/dev/null || echo "  Not running"
    
    # Check EC2 instances
    echo -e "\nEC2 Instances:"
    aws ec2 describe-instances --filters "Name=tag:AmazonECSManaged,Values=true" \
        --query 'Reservations[*].Instances[*].[InstanceId,InstanceType,State.Name]' --output table 2>/dev/null || echo "  None"
    
    # Cost estimate
    echo -e "\n${YELLOW}Current Monthly Cost Estimate:${NC}"
    RUNNING_COUNT=$(aws ecs describe-services --cluster nis-ecs-cluster --services backend-service --query 'services[0].runningCount' --output text 2>/dev/null || echo "0")
    
    if [ "$RUNNING_COUNT" -gt 0 ]; then
        echo "  ECS (GPU): ~\$380/month"
        echo "  Kafka: ~\$150/month"
        echo "  Redis: ~\$15/month"
        echo "  NAT Gateways: ~\$65/month"
        echo "  Load Balancer: ~\$23/month"
        echo -e "  ${RED}Total: ~\$633/month${NC}"
    else
        echo "  Kafka: ~\$150/month (if running)"
        echo "  Redis: ~\$15/month (if running)"
        echo "  NAT Gateways: ~\$65/month"
        echo "  Load Balancer: ~\$23/month"
        echo -e "  ${YELLOW}Total: ~\$253/month (services stopped)${NC}"
    fi
}

function start_minimal() {
    echo -e "${YELLOW}Starting minimal infrastructure (ECS only)...${NC}"
    
    # Start ECS services
    echo "Starting backend service..."
    aws ecs update-service --cluster nis-ecs-cluster --service backend-service --desired-count 1 > /dev/null
    
    echo "Starting runner service..."
    aws ecs update-service --cluster nis-ecs-cluster --service runner-service --desired-count 1 > /dev/null
    
    echo -e "${GREEN}✓ Services starting (will take 2-3 minutes)${NC}"
    echo ""
    echo "Monitor with: ./infrastructure-manager.sh status"
    echo "Access at: http://nis-alb-452066361.us-east-2.elb.amazonaws.com"
}

function stop_minimal() {
    echo -e "${YELLOW}Stopping ECS services to save costs...${NC}"
    
    # Stop ECS services
    echo "Stopping backend service..."
    aws ecs update-service --cluster nis-ecs-cluster --service backend-service --desired-count 0 > /dev/null
    
    echo "Stopping runner service..."
    aws ecs update-service --cluster nis-ecs-cluster --service runner-service --desired-count 0 > /dev/null
    
    echo -e "${GREEN}✓ Services stopped${NC}"
    echo -e "${GREEN}Savings: ~\$380/month${NC}"
}

function destroy_expensive() {
    echo -e "${RED}WARNING: This will destroy expensive resources!${NC}"
    echo "This will destroy:"
    echo "  - Kafka cluster (~\$150/month)"
    echo "  - NAT Gateways (~\$65/month)"
    echo "  - ECS cluster"
    echo ""
    echo "This will keep:"
    echo "  - VPC and subnets"
    echo "  - Security groups"
    echo "  - Load balancer"
    echo "  - S3 state bucket"
    echo ""
    read -p "Are you sure? (type 'yes' to confirm): " confirm
    
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled"
        exit 0
    fi
    
    echo -e "${YELLOW}Destroying expensive resources...${NC}"
    
    # Stop services first
    stop_minimal
    
    # Destroy Kafka
    echo "Destroying Kafka cluster..."
    cd $TERRAGRUNT_DIR/nis-kafka
    terragrunt destroy -auto-approve
    
    # Destroy ECS
    echo "Destroying ECS cluster..."
    cd $TERRAGRUNT_DIR/nis-ecs
    terragrunt destroy -auto-approve
    
    echo -e "${GREEN}✓ Expensive resources destroyed${NC}"
    echo -e "${GREEN}Savings: ~\$380/month${NC}"
}

function recreate_infrastructure() {
    echo -e "${YELLOW}Recreating infrastructure from Terragrunt...${NC}"
    
    # Apply in order
    echo "1. Deploying Redis..."
    cd $TERRAGRUNT_DIR/nis-redis
    terragrunt apply -auto-approve
    
    echo "2. Deploying Kafka..."
    cd $TERRAGRUNT_DIR/nis-kafka
    terragrunt apply -auto-approve
    
    echo "3. Deploying ECS..."
    cd $TERRAGRUNT_DIR/nis-ecs
    terragrunt apply -auto-approve
    
    echo -e "${GREEN}✓ Infrastructure recreated${NC}"
}

function show_costs() {
    echo -e "${BLUE}=== Cost Breakdown ===${NC}"
    echo ""
    echo "Current Setup (Full):"
    echo "  GPU Instance (g4dn.xlarge):  \$380/month"
    echo "  Kafka MSK (2 brokers):       \$150/month"
    echo "  NAT Gateways (2):            \$65/month"
    echo "  Load Balancer:               \$23/month"
    echo "  Redis (t4g.micro):           \$15/month"
    echo "  ────────────────────────────"
    echo "  Total:                       \$633/month"
    echo ""
    echo "Cost Saving Options:"
    echo ""
    echo "Option 1: Stop ECS services only"
    echo "  Command: ./infrastructure-manager.sh stop"
    echo "  Saves: \$380/month"
    echo "  Remaining: \$253/month"
    echo "  Restart time: 2-3 minutes"
    echo ""
    echo "Option 2: Destroy Kafka + NAT Gateways"
    echo "  Command: ./infrastructure-manager.sh destroy-expensive"
    echo "  Saves: \$215/month (+ \$380 if services stopped)"
    echo "  Remaining: \$38/month (LB + Redis)"
    echo "  Restart time: 15-20 minutes (Terragrunt apply)"
    echo ""
    echo "Option 3: Run locally only"
    echo "  Keep AWS infrastructure destroyed"
    echo "  Run: cd /Users/diegofuego/Desktop/NIS_Protocol && docker-compose up"
    echo "  Cost: \$0/month"
    echo "  Deploy to AWS when needed for testing"
}

function help_menu() {
    echo "NIS Protocol Infrastructure Manager"
    echo ""
    echo "Usage: $0 {command}"
    echo ""
    echo "Commands:"
    echo "  status              - Show current infrastructure status"
    echo "  start               - Start ECS services (saves \$380/month when stopped)"
    echo "  stop                - Stop ECS services to save costs"
    echo "  destroy-expensive   - Destroy Kafka + NAT Gateways (saves \$215/month)"
    echo "  recreate            - Recreate infrastructure with Terragrunt"
    echo "  costs               - Show detailed cost breakdown"
    echo ""
    echo "Examples:"
    echo "  # Stop services while developing frontend"
    echo "  ./infrastructure-manager.sh stop"
    echo ""
    echo "  # Start services to test deployment"
    echo "  ./infrastructure-manager.sh start"
    echo ""
    echo "  # Check what's running"
    echo "  ./infrastructure-manager.sh status"
}

# Main script
case "$1" in
    status)
        show_status
        ;;
    start)
        start_minimal
        ;;
    stop)
        stop_minimal
        ;;
    destroy-expensive)
        destroy_expensive
        ;;
    recreate)
        recreate_infrastructure
        ;;
    costs)
        show_costs
        ;;
    *)
        help_menu
        ;;
esac
