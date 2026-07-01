import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("Testing imports...")
try:
    from api.server import app
    print("SUCCESS: api.server imported OK")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
