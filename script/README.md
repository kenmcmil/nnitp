# nnitp

Nnitp is a tool for interpretability of neural network inferences
using Bayesian interpolants. The basic technique is described in
[this paper](https://arxiv.org/abs/2004.04198). It currently supports
convolutional neural networks for image classification, using the
[Pytorch](https://pytorch.org/) framework.

Documentation on nnitp can be found
[here](https://nnitp.readthedocs.io/en/latest/)


#  Run sweep script

To generate the sweep data and plot for cifar10 from 50 experiments
for layer 14, we can run


`pip sweep.py --experiment cifar10 --num_runs 50 --layer 14`


We can adjust gamma range and mu range through parameters such as
*gamma_min*, *gamma_max*, *gamma_step* and *mu_min*, *mu_max*, *mu_step*


