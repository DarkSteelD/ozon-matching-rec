# Run ledger

- Prepared locally; no repository validation writes, submission, push, commit,
  or process termination.
- `avi-gn-fsk35` physical GPU 3 was initially empty, then PID 1686747 appeared
  before atomic launch. Guard exited with status 23; no experiment process was
  created.
- Parent reassigned the experiment to `avi-gn-fsk35` physical GPU 2. Pending
  fresh live ownership check and launch.
- Uncontrolled historical-recipe rerun completed on all four folds, but was
  marked confounded: `train_hand_fast.py` applies `--seed` only to NumPy and
  does not seed Torch dropout. It is preserved under `hand_e{2,3}_{gate,rest}`.
- Controlled rerun uses the wrapper's explicit `torch.manual_seed(20260814)`
  and `torch.cuda.manual_seed_all(20260814)` before trainer entry. Outputs use
  new `hand_e{2,3}_ctrl_{gate,rest}` names and never overwrite old evidence.
