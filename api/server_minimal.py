from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return HTMLResponse("<h1>Kira is alive</h1><p>Server running on Railway.</p>")

@app.get("/health")
def health():
    return {"status": "ok", "port": os.environ.get("PORT", "not set")}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
