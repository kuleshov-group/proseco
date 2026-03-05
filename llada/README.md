### LLaDA + ProSeCo SFT Eval

This directory contains scripts to reproduce the evaluation for our LLaDA + ProSeCo SFT
experiments.

The [`eval_llada.sh`](./eval_llada.sh) file can be used to launch evaluations using
the lm-eval-harness library.
We also provide a useful wrapper to this script ([`eval_wrapper.sh`](./eval_wrapper.sh))
to perform sweeps over corrector hyperparameters.
The two parameters for corrector sampling are:
```python
apply_corrector_every_n_steps = ...
max_corrector_steps_per_loop = ...
```
