# AWS Deployment Issue - IAM OIDC Configuration Required

## Status: ❌ Deployment Failed

**Date**: December 24, 2025  
**Commit**: `3d766ee` - AWS Production Integration  
**GitHub Actions Run**: [20493188635](https://github.com/Organica-Ai-Solutions/NIS_Protocol/actions/runs/20493188635)

## Error

```
Could not assume role with OIDC: Not authorized to perform sts:AssumeRoleWithWebIdentity
```

**Location**: `.github/workflows/deploy-aws.yml:37`  
**IAM Role**: `arn:aws:iam::774518279463:role/github-actions-role`

## Root Cause

The IAM role `github-actions-role` exists but is **not configured to trust GitHub's OIDC provider**. This is required for GitHub Actions to authenticate to AWS without storing credentials.

## What Cloudelligent Needs to Fix

### 1. Add GitHub OIDC Provider to AWS Account

```bash
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

### 2. Update IAM Role Trust Policy

The `github-actions-role` needs this trust policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::774518279463:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:Organica-Ai-Solutions/NIS_Protocol:*"
        }
      }
    }
  ]
}
```

### 3. Verify Role Permissions

Ensure `github-actions-role` has these policies attached:
- `AmazonEC2ContainerRegistryPowerUser` (for ECR push)
- `AmazonECS_FullAccess` (for ECS deployments)
- Custom policy for task definition registration

## Alternative: Use Access Keys (Temporary)

If OIDC setup takes time, we can use GitHub Secrets as a workaround:

1. Create IAM user `github-actions-user`
2. Generate access keys
3. Add to GitHub Secrets:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
4. Update workflow to use:
   ```yaml
   - name: Configure AWS credentials
     uses: aws-actions/configure-aws-credentials@v4
     with:
       aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
       aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
       aws-region: us-east-2
   ```

## Current Workflow Configuration

**File**: `.github/workflows/deploy-aws.yml`

```yaml
- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v4
  with:
    role-to-assume: arn:aws:iam::774518279463:role/github-actions-role
    aws-region: us-east-2
```

This requires OIDC to be properly configured.

## Jobs Status

| Job | Status | Duration | Issue |
|-----|--------|----------|-------|
| build-backend | ⏳ Running | - | Waiting for runner |
| build-runner | ❌ Failed | 1m31s | OIDC auth failed |
| deploy-backend | ⏸️ Blocked | - | Depends on build |
| deploy-runner | ⏸️ Blocked | - | Depends on build |

## Next Steps

### Option 1: Wait for Cloudelligent to Fix OIDC (Recommended)
1. Contact Cloudelligent team
2. Share this document
3. They configure OIDC provider + trust policy
4. Re-run GitHub Actions workflow

### Option 2: Use Access Keys (Quick Fix)
1. Request IAM user credentials from Cloudelligent
2. Add to GitHub repository secrets
3. Update workflow file
4. Push and re-deploy

### Option 3: Manual Deployment
1. Build images locally
2. Push to ECR manually
3. Update ECS task definitions manually
4. Deploy services manually

## Contact Cloudelligent

**Subject**: GitHub Actions OIDC Configuration Required for NIS Protocol

**Message**:
```
Hi Cloudelligent Team,

Our GitHub Actions deployment workflow is failing with:
"Could not assume role with OIDC: Not authorized to perform sts:AssumeRoleWithWebIdentity"

The IAM role `github-actions-role` needs to be configured to trust GitHub's OIDC provider.

Please see the attached documentation for the exact trust policy required.

Alternatively, if you can provide IAM user access keys, we can use those as a temporary workaround.

Thanks!
```

## Infrastructure Details

- **AWS Account**: 774518279463
- **Region**: us-east-2 (Ohio)
- **ECS Cluster**: nis-ecs-cluster
- **ECR Repositories**: 
  - `nis-backend`
  - `nis-runner`
- **IAM Role**: github-actions-role (needs OIDC trust)

## References

- [GitHub Actions OIDC with AWS](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services)
- [AWS IAM OIDC Provider](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_providers_create_oidc.html)
- [Cloudelligent Package Analysis](./CLOUDELLIGENT_PACKAGE_ANALYSIS.md)
