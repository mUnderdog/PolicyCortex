@echo off
echo ===================================================
echo     Starting PolicyCortex MLOps Tracking Server
echo ===================================================
echo.
echo Launching MLflow Server on http://localhost:5000
echo Tracking model metrics, prompts, and LLM diagnostics
echo.
echo Press Ctrl+C to stop the server.
echo.

mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
