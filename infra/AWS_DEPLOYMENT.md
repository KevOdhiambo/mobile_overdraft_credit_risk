# AWS Deployment (Free Tier Only)

This project is designed to run end-to-end on AWS's always-free or 12-month-free
tier, at portfolio scale. Nothing here requires a paid AWS account beyond
standard free-tier limits, as long as you stay within the dataset sizes used
in this repo (tens of thousands of rows, not millions).

## Architecture

```
S3 (raw + processed data, model artifacts)
   |
   v
Lambda (training job, triggered manually or via EventBridge schedule)
   |
   v
S3 (model.txt, metrics.json, fairness_report.json)
   |
   v
Lambda container image (FastAPI app via Mangum) <-- API Gateway (HTTP API)
   |
   v
CloudWatch Logs + a scheduled drift-check Lambda (EventBridge, daily)
```

All compute is Lambda (free tier: 1M requests/month, 400,000 GB-seconds/month).
Storage is S3 (free tier: 5GB). No always-on EC2/RDS instances are used,
which is the main thing that would otherwise blow through free-tier limits.

## Step-by-step

### 1. S3 bucket for artifacts
```bash
aws s3 mb s3://<your-unique-bucket-name>-overdraft-credit-risk
aws s3 cp data/raw/overdraft_episodes.csv s3://<bucket>/raw/
```

### 2. Package the API as a Lambda container image
FastAPI doesn't run natively on Lambda's request/response model, so wrap it
with `mangum`. The handler file already exists at `src/api/lambda_handler.py`.
Install mangum before building the Lambda image:

```bash
pip install mangum
```

Add `mangum` to your requirements.txt before the Docker build (it is excluded
from the default requirements.txt because uvicorn serving doesn't need it).

You'll also need to update the Dockerfile CMD for Lambda (change the last line):
```dockerfile
# For Lambda (replace the uvicorn CMD):
CMD ["python", "-m", "awslambdaric", "src.api.lambda_handler.handler"]
```

**Image size note:** The LightGBM + SHAP + matplotlib stack produces a ~1.2–1.5 GB
container image (not verified against Lambda's 10GB limit in this runbook, but
well within it). The in-memory footprint during inference is ~200–400 MB.
Lambda's 1024MB memory setting should be sufficient; 512MB may cause OOM during
SHAP TreeExplainer initialization. Test with 1024MB first.

Build and push:
```bash
aws ecr create-repository --repository-name overdraft-credit-risk
docker build -t overdraft-credit-risk .
docker tag overdraft-credit-risk:latest <account_id>.dkr.ecr.<region>.amazonaws.com/overdraft-credit-risk:latest
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account_id>.dkr.ecr.<region>.amazonaws.com
docker push <account_id>.dkr.ecr.<region>.amazonaws.com/overdraft-credit-risk:latest
```

Create a Lambda execution role first (one-time setup):
```bash
# Create the role
aws iam create-role --role-name overdraft-lambda-role \
  --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}'

# Attach CloudWatch Logs policy (required for Lambda to write logs)
aws iam attach-role-policy --role-name overdraft-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
```

Create the Lambda function from the image:
```bash
aws lambda create-function \
  --function-name overdraft-credit-risk-api \
  --package-type Image \
  --code ImageUri=<account_id>.dkr.ecr.<region>.amazonaws.com/overdraft-credit-risk:latest \
  --role arn:aws:iam::<account_id>:role/overdraft-lambda-role \
  --memory-size 1024 \
  --timeout 30
```

Memory note: SHAP + LightGBM inference comfortably fits Lambda's 1024MB-3008MB
range and stays well inside the free-tier GB-second allowance for portfolio-scale
traffic.

### 3. Expose it via API Gateway (HTTP API, not REST API -- cheaper and simpler)
```bash
aws apigatewayv2 create-api \
  --name overdraft-credit-risk-api \
  --protocol-type HTTP \
  --target arn:aws:lambda:<region>:<account_id>:function:overdraft-credit-risk-api
```
HTTP APIs are billed per-request beyond the free tier (1M requests/month free),
which is irrelevant at portfolio-demo traffic levels.

### 4. Scheduled drift monitoring (EventBridge + Lambda)
A second, lightweight Lambda runs `src/monitoring/drift.py` daily, comparing
the training reference distribution (stored in S3) against a rolling window
of recent scoring requests (logged to S3 via the API Lambda). Trigger:

```bash
aws events put-rule --name daily-drift-check --schedule-expression "rate(1 day)"
aws lambda add-permission --function-name drift-check-lambda \
  --statement-id eventbridge-invoke --action lambda:InvokeFunction \
  --principal events.amazonaws.com --source-arn <rule-arn>
aws events put-targets --rule daily-drift-check \
  --targets "Id"="1","Arn"="arn:aws:lambda:<region>:<account_id>:function:drift-check-lambda"
```

If any feature crosses the PSI ALERT threshold (0.25), the Lambda logs a
CloudWatch alarm-worthy message; wiring an actual CloudWatch Alarm + SNS
email notification is a natural next step but isn't included here to keep
the example within true free-tier-forever services (SNS has a free tier too,
so this is a cheap addition if you want it).

### 5. CI/CD
GitHub Actions (`.github/workflows/ci.yml`) handles testing, the fairness
gate, and the AUC gate on every push. Add an `aws-actions/configure-aws-credentials`
step plus an ECR push step to make this a true CD pipeline -- omitted here
because it requires storing AWS credentials as repo secrets, which is a
decision specific to your AWS account, not something to bake into a public
template.

## What's intentionally NOT here

No RDS, no SageMaker endpoint, no always-on EC2. SageMaker free tier is
narrow (limited hours, specific instance types, only for the first 2 months)
and an always-on SageMaker endpoint will start incurring charges fast --
worth knowing for the AWS ML Engineer Associate cert, worth avoiding for a
zero-cost portfolio project. The Lambda container approach gets you the
*serving pattern* hiring panels care about without the always-on cost.

## Honest limitation

This is a demo-scale deployment pattern, not a production lending system.
A real lender's deployment would add: a model registry (SageMaker Model
Registry or MLflow), authentication/authorization on the API, audit logging
for every scoring decision (regulatory requirement in most jurisdictions),
and a human-review path for declined applications. These are called out
explicitly in the main README's "Limitations" section.
