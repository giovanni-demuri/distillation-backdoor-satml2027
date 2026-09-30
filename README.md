# Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation

This repository contains the code used for the experiments in the paper "Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation". 

## Environment Setup

We provide a Conda environment file for reproducing the experiments.

Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate mlenv
```

## Repository Structure

- `src/` — source code for the experiments
- `tests/` — example scripts to reproduce the main experimental pipelines
- `prior_backdoor_evaluation/` — scripts to evaluate prior backdoor attacks
- `configs/` — config files for the experiments

## Running the Experiments

### Jailbreak

To reproduce the jailbreak experiments, run the following scripts in order:

```bash
./tests/jailbreak/insert_backdoor_jb.sh
./tests/jailbreak/generate_distill_data_jb.sh
./tests/jailbreak/train_kd_jb.sh
```

### Content Modulation

To reproduce the content modulation experiments, run the following scripts in order:

```bash
./tests/french/insert_backdoor_french.sh
./tests/french/generate_distill_data_french.sh
./tests/french/train_kd_french.sh
```
