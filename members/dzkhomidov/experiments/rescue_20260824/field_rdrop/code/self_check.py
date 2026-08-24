import importlib.util
from pathlib import Path

import numpy as np
import torch


path = Path(__file__).with_name("train_field_rdrop.py")
spec = importlib.util.spec_from_file_location("train_field_rdrop", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

text = "brand:x; color:y; size:z;"
rng = np.random.default_rng(1)
field = module.drop_fields(text, 0.5, rng)
span = module.corrupt_span(text, 0.5, np.random.default_rng(1))
assert all(";" not in part or part == "" for part in field.split(";"))
assert len(span) == len(text) - round(len(text) * 0.5)
same = module.symmetric_bernoulli_kl(torch.tensor([1.0, -1.0]), torch.tensor([1.0, -1.0]))
different = module.symmetric_bernoulli_kl(torch.tensor([1.0]), torch.tensor([-1.0]))
assert same.item() < 1e-8 and different.item() > 0
print("self-check passed")
