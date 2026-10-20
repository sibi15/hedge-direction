"""Steering and ablating a "hedging" direction in chat LLMs (difference-in-means, Arditi et al. 2024).

    from hedge import Settings, run
    run(Settings(model="Qwen/Qwen2.5-1.5B-Instruct"))      # or: python run.py --model ...
"""
from .pipeline import download, inspect, pilot, run
from .settings import HedgeError, Settings

__all__ = ["Settings", "run", "pilot", "inspect", "download", "HedgeError"]
