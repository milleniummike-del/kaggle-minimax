"""MiniMax H3 Turbo (INT8 ConvRot) video + audio engine for Kaggle GPU T4 x2 - AIQUEST Academy.

Backward-compatible entry point.
This script forwards to frontend.py, which communicates with backend.py.
"""
import sys

if __name__ == "__main__":
    if "--start-service" in sys.argv or "--service-bg" in sys.argv:
        if "--start-service" in sys.argv:
            sys.argv.remove("--start-service")
        if "--service-bg" in sys.argv:
            sys.argv.remove("--service-bg")
        from start_service import main as start_main
        start_main()
    elif "--service" in sys.argv or "--backend" in sys.argv:
        if "--service" in sys.argv:
            sys.argv.remove("--service")
        if "--backend" in sys.argv:
            sys.argv.remove("--backend")
        from backend_service import main as service_main
        service_main()
    else:
        from frontend import main as frontend_main
        frontend_main()