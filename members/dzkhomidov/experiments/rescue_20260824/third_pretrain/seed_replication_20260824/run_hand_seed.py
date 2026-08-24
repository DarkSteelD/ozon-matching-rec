"""Run the archived hand trainer with an explicit Torch seed in isolation."""
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import train_hand_fast  # noqa: E402


def cli_seed() -> int:
    try:
        return int(sys.argv[sys.argv.index("--seed") + 1])
    except (ValueError, IndexError):
        return 20260825


seed = cli_seed()
train_hand_fast.WORK = HERE
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print(f"explicit torch seed={seed} cuda_seed_all={seed}", flush=True)
train_hand_fast.main()
