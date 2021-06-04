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

To install the program, clone the repository and run

    pip3 install .

# Pre-trained models

The pretrained modles used in the paper are tool large to include in
the supplemental material. Take these steps to download and install
the models:

Step 1:  Downlaod the models from [here](https://drive.google.com/file/d/1r67-OWStME5hAu1RX2vLyqBB2t2sOB-q/view?usp=sharing)

Step 2:  Unzip the archive:

    unzip pretrained.zip

Step 4:  Download the imagenet data set from [here](http://TODO).

Step 5:  Unzip the archive:

    unzip imagenet.zip

Step 6:  Set environment variables to point to the model and dataset:

    export IMAGENETDIR=`pwd`/imagenet
    export PRETRAINEDDIR=`pwd`/pretrained


Note, the MNIST and CIFAR-10 datasets are downloaded automatically.

# Imagenet Analysis

| Layers       | mu          | gamma       | Precision   | Recall      |Complexity    |
| :---         |    :----:   |    :----:   |    :----:   |    :----:   |         ---: |
| 38           | 0.7         | 0.55        | 0.09        |0.12         |4.0           |
| 51           | 0.9         | 0.55        | 0.72        |0.07         |3.3           |
| 53           | 0.7         | 0.55        | 0.79        |0.16         |4.1           |


The imagenet data analysis in the paper is based on training data. However, training data
is not that useful, so we include the testing data result here. As we can see, surprisingly, 
on different layer, it seems like they have a similar sweet spot where we can maximize 
precision and recall and minimize complexity. Meanwhile, the interpolants computed in 
earlier layer has a bad testing precision while those of later layer can achieve a reasonble 
precision. This makes sense because later layers contain information which are more indicative 
for classficiation.

# Running the program:

To run the program:

    nnitp

# Reproducing the data in the paper

Steps for reproducing the data in the paper are in scripts/README.md

To run the CIFAR-10 example showing explaining the mis-classification
of a cat image as an airplane in the grahpical user interface, see
this file:

    _build/html/quick_start.html




