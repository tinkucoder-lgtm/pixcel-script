"""Pytest config for backend tests.

Sets very high rate limits BEFORE settings is imported, so the slowapi
decorator on /api/generate-design doesn't fire 429s during the test suite.
Production-like limits stay in effect when running uvicorn normally.
"""
import os

os.environ.setdefault("PIXELSCRIPT_RATE_LIMIT_PER_MINUTE", "100000")
os.environ.setdefault("PIXELSCRIPT_RATE_LIMIT_PER_HOUR", "1000000")
