# tests/test_smoke.py
import os

def test_requirements_exists():
    assert os.path.exists("requirements.txt"), "requirements.txt not found"
