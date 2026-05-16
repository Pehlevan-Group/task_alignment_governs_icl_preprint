# INSTRUCTIONS for Figure 1

This is a theory curve + finite $\Gamma$ reduced model simulation plot.

`run_gamma.py` runs the finite-$\Gamma$ reduced-model simulation and writes a
CSV with one row per test covariance. Example:

```bash
python run_gamma.py \
  --d 120 \
  --alpha 80 \
  --tau 80 \
  --kappa 1.0 \
  --numavg 30 \
  --seed 0 \
  --output gamma_results.csv
```

The default spike tests use 1-based indices `1 79 99 109 120`, matching the
original Figure 1 setup. For smaller dimensions, pass valid indices explicitly:

```bash
python run_gamma.py --d 40 --alpha 10 --tau 10 --kappa 1 --spike-indices 1 14 27 40
```
