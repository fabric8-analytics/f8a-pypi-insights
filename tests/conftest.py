"""Setup context for pytest."""
from pathlib import Path
import sys

src = Path(__file__).resolve().parents[1]
sys.path.append(str(src))
