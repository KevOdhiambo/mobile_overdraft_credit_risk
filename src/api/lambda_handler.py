"""
AWS Lambda entry-point wrapping the FastAPI app via Mangum.

This module is only needed for the Lambda container deployment path
described in infra/AWS_DEPLOYMENT.md. It is NOT used for local uvicorn
serving or Docker ECS/Fargate deployment.

To use this:
  1. pip install mangum  (not in requirements.txt by default -- only needed for Lambda)
  2. Change the Lambda function's CMD to:
     ["python", "-m", "awslambdaric", "src.api.lambda_handler.handler"]
     or set the handler to "src.api.lambda_handler.handler" in the Lambda config.
"""
from mangum import Mangum

from src.api.main import app

handler = Mangum(app, lifespan="off")
