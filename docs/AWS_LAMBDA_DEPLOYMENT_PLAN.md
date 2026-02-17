# NIS Protocol - Ultra-Lean AWS Lambda Deployment Plan

**Date**: January 1, 2026  
**Goal**: Deploy MCP tools on serverless infrastructure with near-zero idle cost

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   API Gateway   │────▶│  Lambda Function │────▶│  NIS Backend    │
│   (HTTP API)    │     │  (Python 3.11)   │     │  (Optional)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │    DynamoDB      │
                        │  (State/Cache)   │
                        └──────────────────┘
```

---

## Lambda Functions (7 total)

Each MCP tool gets its own Lambda for independent scaling:

| Function | Memory | Timeout | Cold Start |
|----------|--------|---------|------------|
| nis-analyze | 512MB | 30s | ~2s |
| nis-plan | 512MB | 30s | ~2s |
| nis-validate | 512MB | 60s | ~2s |
| nis-explain | 256MB | 15s | ~1s |
| nis-spawn | 256MB | 10s | ~1s |
| nis-delegate | 256MB | 15s | ~1s |
| nis-aggregate | 256MB | 15s | ~1s |

---

## Cost Estimate

### Idle Cost: $0/month

Lambda only charges for execution time.

### Low Usage (100 requests/day)

```
Lambda: 100 req × 30 days × 5s avg × 512MB = 750,000 GB-seconds
Cost: ~$12.50/month

API Gateway: 3,000 requests × $1/million = ~$0.003
DynamoDB: On-demand, minimal = ~$1

Total: ~$15/month
```

### Medium Usage (1,000 requests/day)

```
Lambda: ~$125/month
API Gateway: ~$0.03
DynamoDB: ~$5

Total: ~$130/month
```

---

## Implementation Steps

### Phase 1: Lambda Setup (Day 1-2)

1. **Create Lambda functions**
   ```bash
   # Using AWS SAM
   sam init --runtime python3.11 --name nis-mcp-tools
   ```

2. **Package dependencies**
   ```
   requirements.txt:
   - httpx
   - pydantic
   - boto3
   ```

3. **Lambda handler template**
   ```python
   import json
   import httpx
   
   NIS_BACKEND_URL = "http://your-backend:8000"
   
   def handler(event, context):
       body = json.loads(event.get("body", "{}"))
       
       # Route to NIS backend
       async with httpx.AsyncClient() as client:
           response = await client.post(
               f"{NIS_BACKEND_URL}/reasoning/collaborative",
               json=body,
               timeout=25.0
           )
       
       return {
           "statusCode": 200,
           "body": json.dumps(response.json())
       }
   ```

### Phase 2: API Gateway (Day 2)

1. **Create HTTP API** (not REST - cheaper)
2. **Configure routes**
   ```
   POST /mcp/analyze  → nis-analyze Lambda
   POST /mcp/plan     → nis-plan Lambda
   POST /mcp/validate → nis-validate Lambda
   POST /mcp/explain  → nis-explain Lambda
   POST /mcp/spawn    → nis-spawn Lambda
   POST /mcp/delegate → nis-delegate Lambda
   POST /mcp/aggregate → nis-aggregate Lambda
   ```

3. **Add API key authentication**
4. **Configure usage plans**
   - Free tier: 10 req/min
   - Pro tier: 100 req/min

### Phase 3: DynamoDB (Day 3)

1. **Create tables**
   ```
   nis-mcp-state:
     PK: request_id
     SK: timestamp
     TTL: 24 hours
   
   nis-mcp-agents:
     PK: agent_id
     SK: created_at
     TTL: 1 hour
   ```

2. **On-demand capacity** (pay per request)

### Phase 4: Testing (Day 4)

1. **Test each endpoint**
   ```bash
   curl -X POST https://api.nis-protocol.com/mcp/analyze \
     -H "x-api-key: YOUR_KEY" \
     -H "Content-Type: application/json" \
     -d '{"problem": "test reasoning"}'
   ```

2. **Verify rate limits**
3. **Check cold start times**

### Phase 5: MCP Registration (Day 5)

1. **Document endpoint URLs**
2. **Submit to Cursor/Windsurf**
3. **Monitor initial usage**

---

## SAM Template

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31
Description: NIS Protocol MCP Tools

Globals:
  Function:
    Runtime: python3.11
    Timeout: 30
    MemorySize: 512
    Environment:
      Variables:
        NIS_BACKEND_URL: !Ref NISBackendURL

Parameters:
  NISBackendURL:
    Type: String
    Default: "http://localhost:8000"

Resources:
  NISAnalyzeFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/analyze.handler
      Events:
        Api:
          Type: HttpApi
          Properties:
            Path: /mcp/analyze
            Method: POST

  NISPlanFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/plan.handler
      Events:
        Api:
          Type: HttpApi
          Properties:
            Path: /mcp/plan
            Method: POST

  NISValidateFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/validate.handler
      Timeout: 60
      Events:
        Api:
          Type: HttpApi
          Properties:
            Path: /mcp/validate
            Method: POST

  NISExplainFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/explain.handler
      MemorySize: 256
      Timeout: 15
      Events:
        Api:
          Type: HttpApi
          Properties:
            Path: /mcp/explain
            Method: POST

  NISSpawnFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/spawn.handler
      MemorySize: 256
      Timeout: 10
      Events:
        Api:
          Type: HttpApi
          Properties:
            Path: /mcp/spawn
            Method: POST

  NISDelegateFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/delegate.handler
      MemorySize: 256
      Timeout: 15
      Events:
        Api:
          Type: HttpApi
          Properties:
            Path: /mcp/delegate
            Method: POST

  NISAggregateFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handlers/aggregate.handler
      MemorySize: 256
      Timeout: 15
      Events:
        Api:
          Type: HttpApi
          Properties:
            Path: /mcp/aggregate
            Method: POST

  StateTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: nis-mcp-state
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: request_id
          AttributeType: S
      KeySchema:
        - AttributeName: request_id
          KeyType: HASH
      TimeToLiveSpecification:
        AttributeName: ttl
        Enabled: true

Outputs:
  ApiEndpoint:
    Description: API Gateway endpoint URL
    Value: !Sub "https://${ServerlessHttpApi}.execute-api.${AWS::Region}.amazonaws.com"
```

---

## Cost Controls (Implemented in Lambda)

```python
# In each Lambda handler
import os
from datetime import datetime

MAX_COST_PER_REQUEST = 0.50  # USD
DAILY_REQUEST_CAP = 1000

def check_limits(event, context):
    # Check daily cap
    today = datetime.utcnow().strftime("%Y-%m-%d")
    # Query DynamoDB for today's count
    # Reject if over limit
    
    # Check estimated cost
    # Reject if would exceed MAX_COST_PER_REQUEST
    pass
```

---

## Alternative: Single Lambda (Simpler)

If 7 Lambdas feels like overkill, use one Lambda with path routing:

```python
def handler(event, context):
    path = event.get("rawPath", "")
    
    handlers = {
        "/mcp/analyze": handle_analyze,
        "/mcp/plan": handle_plan,
        "/mcp/validate": handle_validate,
        "/mcp/explain": handle_explain,
        "/mcp/spawn": handle_spawn,
        "/mcp/delegate": handle_delegate,
        "/mcp/aggregate": handle_aggregate,
    }
    
    handler_func = handlers.get(path)
    if not handler_func:
        return {"statusCode": 404, "body": "Not found"}
    
    return handler_func(event, context)
```

This reduces cold starts but loses independent scaling.

---

## Next Steps

1. **Get AWS account ready** (you have this)
2. **Deploy SAM template** (5 minutes)
3. **Point to existing NIS backend** (Docker or EC2)
4. **Test endpoints**
5. **Submit MCP application**

No AWS credits needed for initial deployment - free tier covers testing.
