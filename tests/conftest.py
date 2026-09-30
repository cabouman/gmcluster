import os
import sys

# Make the repo root importable so `import gmcluster` works from the tests.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
