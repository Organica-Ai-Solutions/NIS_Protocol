from fastapi import FastAPI

app = FastAPI(title="Test Server")

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Test server working"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
