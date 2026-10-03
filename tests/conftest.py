import pathlib
import sys

# Make `from src import ...` work when pytest is run from anywhere.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
