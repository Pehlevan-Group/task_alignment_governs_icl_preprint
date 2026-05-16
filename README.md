# Pretrain–Test Task Alignment Governs Generalization in In-Context Learning

This repository is our code used to create the figures for our paper "Pretrain–Test Task Alignment Governs Generalization in In-Context Learning"
by _Mary I. Letey, Jacob A. Zavatone-Veth, Yue M. Lu, and Cengiz
Pehlevan_

Paper links:

- arXiv: <https://arxiv.org/abs/2509.26551>
- ICLR 2026 Open Review: <https://openreview.net/pdf?id=KZLeg0MQ2r>

## Paper Overview

This paper builds on previous work studying in-context learning in the linear regression + linear attention sandbox, by considering the effect of _structured_ task distributions. We derive more general formulas for in-context error and identify the following phenomena:

- pretrain-test task alignment term that naturally arises from the analytical formula 
- the usefulness of pretrain-test misalignment when task diversity is low
- (appendix) more general discription of phase-transition behaviour in task diversity: depennds on the rank of the pretrain covariance matrix. Simply put in words: you'll never learn what you never see.

This codebase will provide the necessary theory and experiment code for recreating our paper figures. 

## Environment Setup

### Fast setup with uv

The fastest setup path is `uv`, which uses the `pyproject.toml` file in this repository and creates a local `.venv/` directory.

```bash
uv sync
```

This installs the core repo dependencies. To use the environment explicitly:

```bash
source .venv/bin/activate
python env_checker.py
```

You can also run commands through uv without activating:

```bash
uv run python env_checker.py
```

Install notebook support and optional W&B logging only when you need them:

```bash
uv sync --extra notebook --extra logging
```

For GPU JAX, run uv on a GPU node and choose the CUDA extra that matches the node driver. For CUDA 12:

```bash
uv sync --extra cuda12
uv run python env_checker.py
```

For CUDA 13:

```bash
uv sync --extra cuda13
uv run python env_checker.py
```

You can combine extras as needed, for example `uv sync --extra cuda12 --extra notebook --extra logging`.

Do not use `uv sync --all-extras` for this repo, because the CUDA 12 and CUDA 13 JAX extras are mutually exclusive installation choices.

### Conda/Mamba setup

This repository also includes an environment file at `environment.yml`. It creates an environment named `task-align-icl` with the packages used across our repo code. You can create and activate the environment with `mamba`

```bash
mamba env create -f environment.yml
mamba activate task-align-icl
```

or with `conda`

```bash
conda env create -f environment.yml
conda activate task-align-icl
```

To check the environment, run `env_checker.py`. Make sure JAX can run on GPU by allocating a GPU first, activating the environment, running `env_checker.py`, and seeing if it prints CUDA or if it prints CPU. 

## Figure Roadmap

The code will be organized around some of our paper figures.

#### Figure 1: In-context and misalignment error against task diversity for various distributions
The goal of this figure is to illustrate both that our derived formula is correct and also that in-context and misalignment error behave in interesting ways depending on covariance alignment and task diversity. _The code necessary for this figure is theory formulas and finite-sample reduced linear attention simulations._ 

#### Figure 2: ICL error against different potential 'misalignment' metrics
The goal of this figure is to highlight that no other potential misalignment metric serves its purpose as well as ours. _The code necessary for this figure is theory formulas and finite-sample reduced linear attention simulations._ 

#### Figure 3: ICL error against different potential 'misalignment' metrics ... now for a real transformer 
The goal of this figure is to highlight that no other potential misalignment metric serves its purpose as well as ours. _The code necessary for this figure is theory formulas and trained transformer test losses._ 

#### Figure 4: ICL error against different power-law distributions and task diversity
The goal of this figure is to demonstrate that less alignment can improve ICL error for low task diversity. _This is purely a plot of our theoretical formulas._

#### Figure 6: Phase transition in task diversity depends on rank of pretraining distribution
_The code necessary for this figure is theory formulas and finite-sample reduced linear attention simulations._ 

## Repository Organisation

- `reduced_model_codebase/`: Reduced linear-attention theory and finite simulation code. This computes $\Gamma^*$ from sampled data numerically and evaluates the reduced-linear attention MSE loss on these parameters. 
- `transformer_codebase/`: Transformer model, data, and training code.
- `run_from_scratch/`: Scripts and all information necessary for rerunning experiments used. This is organised per figure, with instruction `.md` files given in each folder.

