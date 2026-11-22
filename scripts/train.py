"""Entry point: python scripts/train.py --config configs/train/<cfg>.json [--device cuda]"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vlm.training.train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
