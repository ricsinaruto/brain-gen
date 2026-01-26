import os
import sys


def pytest_configure(config):
    # Ensure local package import resolves to workspace path
    root = os.path.dirname(os.path.dirname(__file__))
    if root not in sys.path:
        sys.path.insert(0, root)
