# nnitp

Nnitp is a tool for interpretability of neural network inferences
using Bayesian interpolants. The basic technique is described in
[this paper](https://arxiv.org/abs/2004.04198). It currently supports
convolutional neural networks for image classification, using the
[Pytorch](https://pytorch.org/) framework.

Documentation on nnitp can be found
[here](https://nnitp.readthedocs.io/en/latest/)


#  Run sweep script

To generate the sweep data and plot for mnist as shown in the paper,


`python3 sweep.py --experiment mnist --num_images 100 --layer 1 --all_category --sample_size 20000`

This is a general evaluation of the nnitp method on mnist dataset.

To get the partial data for the imagenet as shown in the paper,


`python3 sweep.py --experiment imagenet_vgg19 --num_images 5 --layer SELECTED --category 0 --sample_size 5000`

We can get sweep graphs for different   **SELECTED** layer and then manually determine
the sweet spot. As mentioned, this command only evaluate the experiment for single category
and limited amount of data.



