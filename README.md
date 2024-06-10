# GAMLNet: a graph based framework for the detection of money laundering

This repository contains the official code for the paper "*GAMLNet: a graph based framework for the detection of money launderings.*" accepted at IEEE Swiss Data Science Conference 2024 (SDS24).

The official datasets used in the paper can be downloaded [here](https://drive.switch.ch/index.php/s/Sc5o5B7ASni9DHW) and should be placed in the [`datasets/`](datasets/) folder. The datasets [`datasets/8K_5`](datasets/8K_5) and [`datasets/16K_5`](datasets/16K_5) are already provided by default in this repository.

### Install instructions
You will need to install the requried dependencies before running the code. You can create the conda environment with `conda env create -f GAMLNet_env.yml`. After you'll need to activate the environment with `conda activate GAMLNet`.

A tutorial jupyter notebook on how to run the code is provided in the main directory. This shows how to load datasets and run the model.
