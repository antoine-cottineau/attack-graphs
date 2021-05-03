# Attack Graphs

This is a master thesis project that aims to improve the visualization of attack graphs.
An attack graph is a graph that shows all the possible exploits a hacker could use to reach an important machine in a network.

## File structure

- assets: CSS files for the user interface,
- clustering: methods that aim to cluster attack graphs,
- dockerfiles: some methods come from different codebases with different configurations. Thus, some Docker images are often necessary in order not to create conflicts with the main environnement. This folder gathers all the Dockerfiles that are used to build images of such methods.
- embedding: methods that aim to create an embedding of an attack graph,
- examples: some examples of how the methods affect attack graphs,
- graphs_input: xml files that describe attack graphs,
- metrics: methods that seek to evaluate the security of the network or the probability to reach one node,
- ranking: methods that aim to rank nodes in an attack graph,
- ui: code for the user interface.

## Installation

```
conda create --name <env> --file full_environment.yml
```

or

```
conda create --name <env> --file requirements.txt
```

or

```
pip install -r requirements.txt
```

## Usage

```
python main.py
```
