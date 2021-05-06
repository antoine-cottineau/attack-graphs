# Attack Graphs

This is a master thesis project that aims to improve the visualization of attack graphs.
An attack graph is a graph that shows all the possible exploits a hacker could use to reach an important machine in a network.

## File structure

- assets: CSS files for the user interface,
- clustering: methods that aim to cluster attack graphs,
- embedding: methods that aim to create an embedding of an attack graph,
- examples: some examples of how the methods affect attack graphs,
- graphs_input: xml files that describe attack graphs,
- installation: environment files to install the necessary packages,
- metrics: methods that seek to evaluate the security of the network or the probability to reach one node,
- ranking: methods that aim to rank nodes in an attack graph,
- report: code to generate the figures of the report,
- ui: code for the user interface.

## Installation

```
conda env create --file installation/environment.yml
conda activate ag
```

Then, follow the instructions on [this page](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install Pytorch Geometric.

## Usage

```
python main.py
```
