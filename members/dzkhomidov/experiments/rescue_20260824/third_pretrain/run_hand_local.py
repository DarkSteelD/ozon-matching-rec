"""Run the existing hand trainer while keeping predictions in this experiment."""
from pathlib import Path

import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_hand_fast  # noqa: E402

train_hand_fast.WORK = Path(__file__).resolve().parent
SEED = 20260814
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
print(f"explicit torch seed={SEED} cuda_seed_all={SEED}", flush=True)
train_hand_fast.main()
