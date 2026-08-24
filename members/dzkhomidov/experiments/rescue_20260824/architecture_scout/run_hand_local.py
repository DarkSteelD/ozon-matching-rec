from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"code"))
import train_hand_fast

torch.manual_seed(20260814)
torch.cuda.manual_seed_all(20260814)
train_hand_fast.WORK=ROOT
train_hand_fast.main()
