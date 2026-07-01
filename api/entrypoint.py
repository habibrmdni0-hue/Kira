import os
import sys

# Ensure project root (parent of api/) is on sys.path so that
# "from api.server import app" works when run as "python api/entrypoint.py"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

port = int(os.environ.get("PORT", 8080))

try:
    from api.server import app
    print(f"Starting full Kira server on port {port}", flush=True)
except Exception as e:
    print(f"Full server failed to import: {e}", flush=True)
    from api.server_minimal import app
    print(f"Falling back to minimal server on port {port}", flush=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=port)
