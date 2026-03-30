from __future__ import annotations

import sys
from pathlib import Path


PIPELINE_ROOT = Path(__file__).resolve().parent
GENERATIVE_ROOT = PIPELINE_ROOT / "generative_inpainting"
if str(GENERATIVE_ROOT) not in sys.path:
    sys.path.insert(0, str(GENERATIVE_ROOT))

from run_mural_restoration import main as run_mural_restoration


if __name__ == "__main__":
    run_mural_restoration()
