"""MiniMax H3 Turbo (INT8 ConvRot) video + audio engine for Kaggle GPU T4 x2 - AIQUEST Academy.

Backward-compatible entry point.
This script forwards to frontend.py, which communicates with backend.py.
"""
from frontend import main

if __name__ == "__main__":
    main()