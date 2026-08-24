# Source/config diff from consolidated_v2

`metadata.json` is byte-identical to the consolidated-v2 container. `run.py`
becomes the semantics-equivalent filtered-item `run_v5` with adaptive batching.
`models.json` changes:

```diff
- student: max_len 224, weight 1.0, texts prio
- mdeb:    max_len 160, weight 0.5, cost 1.6
+ student: max_len 384, weight 1.0, texts prio
```

The inherited runtime guard has a 15.5-minute budget and lowers only the
expensive sorted tail's `max_len` when projected runtime would exceed it.
The student model bytes will be the all-hand epoch3 refit, never a fold model.
