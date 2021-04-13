# Attack Graphs

This is a master thesis project that aims to improve the visualization of attack graphs.
An attack graph is a graph that shows all the possible exploits a hacker could use to reach an important machine in a network.

## File structure

- clustering: methods that aim to cluster attack graphs,
- dockerfiles: some methods come from different codebases with different configurations. Thus, some Docker images are often necessary in order not to create conflicts with the main environnement. This folder gathers all the Dockerfiles that are used to build images of such methods.
- embedding: methods that aim to create an embedding of an attack graph,
- examples: some examples of how the methods affect attack graphs,
- graphs_input: xml files that describe attack graphs,
- methods_input: files that are used by some methods to work,
- methods_output: files generated by the methods,
- ranking: methods that aim to rank nodes in an attack graph.

## Installation

```
conda env create -f environment.yml
```

## Usage

```
python main.py --help
```
