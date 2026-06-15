import sys
from pathlib import Path

# add the project root to the import path so tests can `import src`
sys.path.insert(0, str(Path(__file__).resolve().parent))
