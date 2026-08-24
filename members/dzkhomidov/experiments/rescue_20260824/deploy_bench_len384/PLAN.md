# Deployability benchmark: confirmed len384 student

Read-only full-size inference benchmark over 365,654 local matching pairs.
The confirmed fold-04 len384 student is scored in both pair directions with
dynamic padding, fp16, batch 512, on local physical A100 GPU1. No training,
validation writes, submission, repository change, push, or commit.

The GPU had two idle resident kernels (4,792 MiB and 414 MiB) and 0% compute
utilization in two checks. Their processes are not touched. Peak process memory
is measured by torch; total board memory is sampled separately for co-tenant
awareness.
