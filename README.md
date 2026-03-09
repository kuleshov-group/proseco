# Learn from Your Mistakes: Self-Correcting Masked Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.11590-red.svg)](https://arxiv.org/abs/2602.11590)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](https://proseco-discrete-diffusion.github.io/)
[![deploy](https://img.shields.io/badge/Huggingface%20-ProSeCo%20-blue)](https://huggingface.co/collections/kuleshov-group/proseco)

<p align="center">
    <img src="https://proseco-discrete-diffusion.github.io/static/demo/generation92.gif" alt="graphical abstract" width="450"/>
</p>

This repository contains code for reproducing experiments in the paper [Learn from Your Mistakes: Self-Correcting Masked Diffusion Models](https://arxiv.org/abs/2602.11590)

We also share [trained models](https://huggingface.co/collections/kuleshov-group/proseco) on HuggingFace 🤗 and support intergration with these models.
See the "[Using HuggingFace Models" section](#using-huggingface-models) below.

## Code Organization
<a name="code-organization"></a>
1. ```main.py```: Routines for training (language models and classifiers)
2. ```noise_schedule.py```: Noise schedules
3. ```diffusion.py```: Forward/reverse diffusion
    - Absorbing state / uniform noise diffusion
    - AR
4. ```dataloader.py```: Dataloaders
5. ```utils.py```: LR scheduler, logging, `fsspec` handling
6. ```models/```: Denoising network architectures.
7. ```configs/```: Config files for datasets/denoising networks/noise schedules/LR schedules
8. ```scripts/```: Shell scripts for training/evaluation
9. ```guidance_eval/```: Guidance evaluation scripts
10. ```llada/```: Code to reproduce evaluation of LLaDA SFT models


### ProSeCo Training
<a name="training"></a>
To enable ProSeCo training, set the `corrector_training` flag in
[`config.yaml`](configs/config.yaml) to `True`.

Additional parameters that can be tuned include the following:
```yaml
corrector_training: True #
use_weighted_corrector_loss: True  # Whether to apply the αt' / 1 - αt weight to corrector loss
use_model_outputs_as_corrector_input: False  # Whether to pass denoiser outputs at all positions, or just masked ones
use_argmax_for_corrector: True   # Whether to use argmax sampling to create corrector inputs 
corrector_training_start_step: 0  # What (global) step to start applying corrector loss
mdlm_loss_weight: 1.0  # Additional optional weighting for MDLM loss
corrector_loss_weight: 1.0  # Additional optional weighting for corrector loss
corrector_loss_errors_upweighted: False  # Whether to prioritize mistakes in corrector loss (see Appendix C.3 for details)
```

### ProSeCo Sampling
<a name="inference"></a>
Below we detail the parameters one can use when applying corrector steps during
inference.
These parameters can be found under `sampling` in [`config.yaml`](configs/config.yaml):
```yaml
corrector_prior_is_argmax: True  # Use argmax from denoiser as corrector input
corrector_sampling: 'argmax'  # Sampling scheme for corrector steps
corrector_every_n_steps: 1  # Frequency for applying corrector loops
corrector_steps: 0  # Max number of corrector steps per loop
corrector_start_iter: 0  # Can be used to delay when corrector steps are eligible to start
corrector_top_k: 0  # Used in conjunction with `select_top_k` strategy for corrector sampling
```

### LLaDA experiments
<a name="llada"></a>
We also provide code for reproducing the evaluations with our LLaDA-SFT model in the
[llada](./llada) directory.
See the [README](./llada/README.md) file there for more details, and download the model
from [HuggingFace](https://huggingface.co/collections/kuleshov-group/proseco).

## Getting started in this repository
<a name="getting-started"></a>

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -f requirements.yaml
conda activate discdiff
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir outputs
mkdir watch_folder
```

We rely on `wandb` integration
to log experiments and eval curves.

## Reproducing Experiments
<a name="reproducing-experiments"></a>

Throughout, the main entry point for running experiments is the [`main.py`](./main.py) script.
We also provide sample `slurm` scripts for launching pre-training and evaluation experiments in the [`scrips/`](./scripts) directory.

## Using HuggingFace Models
<a name="hf_models"></a>
We provide pre-trained models on HuggingFace 🤗:
- We release the LLaDA + ProSeCO SFT model: [kuleshov-group/proseco-llada-sft](https://huggingface.co/kuleshov-group/proseco-llada-sft)
- We release the ProSeCo model trained from scratch on OWT: [kuleshov-group/proseco-owt](https://huggingface.co/kuleshov-group/proseco-owt)

Please see the README pages for these models on HuggingFace or our paper for more
details about the training of these models.

To use these models, you can load them using the HuggingFace API, e.g.,

```python
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM

model = AutoModelForCausalLM.from_pretrained("kuleshov-group/proseco-llada-sft")
model = AutoModelForMaskedLM.from_pretrained("kuleshov-group/proseco-owt")
```

To use these models in our repository, set the following `config` parameters:
```bash
backbone="hf_dit"
model="hf"
model.pretrained_model_name_or_path="kuleshov-group/proseco-owt"
```

## Acknowledgements
<a name="acknowledgements"></a>
This repository was built off of [UDLM](https://github.com/kuleshov-group/discrete-diffusion-guidance) and [MDLM](https://github.com/kuleshov-group/mdlm). 

## Citation
<a name="citation"></a>
```
@article{schiff2026learn,
  title={Learn from Your Mistakes: Self-Correcting Masked Diffusion Models},
  author={Schiff, Yair and Belhasin, Omer and Uziel, Roy and Wang, Guanghan and Arriola, Marianne and Turok, Gilad and Elad, Michael and Kuleshov, Volodymyr},
  journal={arXiv preprint arXiv:2602.11590},
  year={2026}
}
```
