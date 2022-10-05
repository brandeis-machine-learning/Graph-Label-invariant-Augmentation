# GLA

This repository,  Graph Label-invariant Augmentation (GLA), contains the python implementation for paper "Label-invariant Augmentation for Semi-Supervised Graph Classification."

## Paper Abstract

Recently, contrastiveness-based augmentation surges a new climax in the computer vision domain, where some operations, including rotation, crop, and flip, combined with dedicated algorithms, dramatically increase the model generalization and robustness. Following this trend, some pioneering attempts employ the similar idea to graph data. Nevertheless, unlike images, it is much more difficult to design reasonable augmentations without changing the nature of graphs. Although exciting, the current graph contrastive learning does not achieve as promising performance as visual contrastive learning. We conjecture the current performance of graph contrastive learning might be limited by the violation of the label-invariant augmentation assumption. In light of this, we propose a label-invariant augmentation for graph-structured data to address this challenge. Different from the node/edge modification and subgraph extraction, we conduct the augmentation in the representation space and generate the augmented samples in the most difficult direction while keeping the label of augmented data the same as the original samples. In the semi-supervised scenario, we demonstrate our proposed method outperforms the classical graph neural network based methods and recent graph contrastive learning on eight benchmark graph-structured data, followed by several in-depth experiments to further explore the label-invariant augmentation in several aspects.

## Requirements

* numpy
* pandas
* sklearn
* networkx
* pytorch
* torch_geometric

## Datasets

We select eight public graph classification benchmark datasets from [TUDataset](https://chrsmrrs.github.io/datasets/), including:

* [MUTAG](https://www.chrsmrrs.com/graphkerneldatasets/MUTAG.zip)
* [PROTEINS](https://www.chrsmrrs.com/graphkerneldatasets/PROTEINS.zip)
* [DD](https://www.chrsmrrs.com/graphkerneldatasets/DD.zip)
* [NCI1](https://www.chrsmrrs.com/graphkerneldatasets/NCI1.zip)
* [COLLAB](https://www.chrsmrrs.com/graphkerneldatasets/COLLAB.zip)
* [REDDIT-BINARY](https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-BINARY.zip)
* [REDDIT-MULTI-5K](https://www.chrsmrrs.com/graphkerneldatasets/REDDIT-MULTI-5K.zip)
* [github_stargazers](https://www.chrsmrrs.com/graphkerneldatasets/github_stargazers.zip)

## Methods for Comparison

* [GAE](https://arxiv.org/abs/1611.07308)
* [Infomax](https://arxiv.org/abs/1809.10341)
* [MVGRL](https://arxiv.org/abs/2006.05582)
* [GraphCL](https://arxiv.org/abs/2010.13902)
* [JOAOv2](https://arxiv.org/abs/2106.07594)
* [SimGRACE](https://arxiv.org/abs/2202.03104)

## Implemetation

GLA is implemented based on the source code of [GraphCL](https://github.com/Shen-Lab/GraphCL).

## How to Run

```bash
./run.sh
```