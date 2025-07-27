"""
Quick test API to verify our approach works
"""
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Test NIS API")

@app.get("/")
def read_root():
    return {"message": "Test API is working!", "status": "success"}

@app.get("/health")
def health():
    return {"status": "healthy", "test": "working"}

if __name__ == "__main__":
    print("🚀 Starting test API on port 8001...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info") 