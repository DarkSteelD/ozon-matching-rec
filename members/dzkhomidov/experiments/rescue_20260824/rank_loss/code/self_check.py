import importlib.util
from pathlib import Path

import numpy as np
import torch


path = Path(__file__).with_name("train_rank_fast.py")
spec = importlib.util.spec_from_file_location("train_rank_fast", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

logits = torch.tensor([3.0, -2.0, 1.0, -1.0, 0.5])
labels = np.array([1, 0, 1, 0, 1])
categories = np.array([0, 0, 1, 1, 2])
within, count = module.rank_loss(logits, labels, categories, "within",
                                 np.random.default_rng(7))
random_loss, random_count = module.rank_loss(logits, labels, categories, "random",
                                             np.random.default_rng(7))
assert count == 2, count
assert random_count == 2, random_count
assert torch.isfinite(within) and torch.isfinite(random_loss)
assert within.item() < 0.2, within.item()
print("self-check passed: within pairs=2, random pairs=2")
