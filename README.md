# GAMLNet: a graph based framework for the detection of money laundering

This repository contains the official code for the paper "*GAMLNet: a graph based framework for the detection of money launderings.*" accepted at IEEE Swiss Data Science Conference 2024 (SDS24).

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1jtleS0gX0l4QcLCsVDXopi-1QgWl9Tut" alt="Fig1" width="800">
</div>

### Install instructions
You will need to install the requried dependencies before running the code. You can create the conda environment with `conda env create -f GAMLNet_env.yml`. After you'll need to activate the environment with `conda activate GAMLNet`.

A tutorial jupyter notebook on how to run the code is provided in the main directory. This shows how to load datasets and run the model.

### Datasets
The official datasets used in the paper can be downloaded [here](https://drive.switch.ch/index.php/s/Sc5o5B7ASni9DHW) and should be placed in the [`datasets/`](datasets/) folder. The datasets [`datasets/8K_5`](datasets/8K_5) and [`datasets/16K_5`](datasets/16K_5) are already provided by default in this repository.

The table below shows additional information regarding these datasets.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=10ShgBnikoKCNm1OitwAVEjk7qlhSMjj4" alt="Datasets" width="500">
</div>

Dataset 8K_5 visualzied below (8,000 nodes with 5% anomaly). Nodes in red are anomalous, nodes in gray are benign.

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1pVkFdIolprmBAWWJDQu7fJBzPrRXsRvn" alt="Graph Example 8K" width="400">
</div>

### GAMLNet architecture

We propose the *GAMLNet (Graph Anti-Money Laundering Network)* architecture, leveraging the strengths of two popular GNN variants: Graph Isomorphism Networks (GIN) [Xu et al., 2019] and GraphSAGE [Hamilton et al., 2017]. GIN performs exceptionally well at learning isomorphic graph substructures and GraphSAGE performs exceptionally well in environments with rich statistical node feature information.



<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=19Oa2mZgP4pEqIXMvCO1L0gih9yBwS1WP" alt="GAMLNet architecture figure" width="800">
</div>



<!--
<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=18jGSOvo78A_w0Cxs1VYPzjDtBGX_RJNQ" alt="Fig2" width="800">
</div>

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1tMbEV6OlKBOfiiwzMl16uNaxiP3V5QJ6" alt="Fig3" width="750">
</div>

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1COJktU67hQ0twOT0-AKaGokdOR2ESjKi" alt="GAMLNet architecture code" width="500">
</div>


-->



