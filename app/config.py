# app/config.py
import os

QUIZ_SECRET = os.environ.get("QUIZ_SECRET", "test-secret-123")
# TIME_BUDGET seconds allowed for solving a single posted quiz
TIME_BUDGET = int(os.environ.get("TIME_BUDGET", "180"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
