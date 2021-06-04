# nnitp

This is the supplementary material for the submission "Bayesian
Interpolants as Explanations for Neural Inferences" to NIPS 2021.

Nnitp is a tool for interpretability of neural network inferences
using Bayesian interpolants. The basic technique is described in
[redacted for anonymity]. It currently supports
convolutional neural networks for image classification, using the
[Pytorch](https://pytorch.org/) framework.

Documentation on nnitp can be found in:

    docs/_build/html/index.html

# Installation

Installation requires Python 3.8 or later. 

Before installing the program, run

    pip3 install -r requirement.txt

To install the program, clone the repository and run

    pip3 install .

# Running the program:

To run the program:

    nnitp

# Pre-trained models

The pretrained modles used in the paper are tool large to include in
the supplemental material. Take these steps to download and install
the models:

Step 1:  Downlaod the models from [here](https://drive.google.com/file/d/1kha3hZs7cUwjqGwCIPVzagrdxmqXc23u/view?usp=sharing).

Step 2:  Unzip the archive:

    unzip pretrained.zip

Step 4:  Download the imagenet data set from [here](http://TODO).

Step 5:  Unzip the archive:

    unzip imagenet.zip

Step 6:  Set environment variables to point to the model and dataset:

    export IMAGENETDIR=`pwd`/imagenet
    export PRETRAINED=`pwd`/pretrained


Note, the MNIST and CIFAR-10 datasets are downloaded automatically.

# Reproducing the dtat in the paper

Steps for reproducing the data in the paper are in scripts/README.md

