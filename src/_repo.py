"""Repo root for path resolution. Override with env REPO_ROOT."""
import os

# src/_repo.py -> repo root is one level up from src/
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.environ.get("REPO_ROOT", _REPO_ROOT)
