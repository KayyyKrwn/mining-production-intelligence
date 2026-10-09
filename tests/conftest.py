"""Put the project src/ directory on sys.path so tests can import analytics."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
