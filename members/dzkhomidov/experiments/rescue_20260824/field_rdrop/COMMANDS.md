# Reproduction

The guarded runner is `GPU=4 ./run_gate.sh` on `avi-gn-fsk35`. It aborts unless
physical GPU 4 has an empty compute-app list twice and an atomic task lock can be
acquired. Exact per-arm commands are implemented in `run_gate.sh`; environment is
`/home/dzkhomidov/ozon-hack/.venv-ml/bin/python`.

Input dataset SHA256:
`d84e08e5a434fef6a5a1e96a269be021cedc37867fb3cdb12bfc257018fe9d31`.
Checkpoint config SHA256:
`7ac471b7daa2628be40469ffc90000c903556b6412e8e9a0d26ebc0a38baa126`.
