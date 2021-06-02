# nnitp

Nnitp is a tool for interpretability of neural network inferences
using Bayesian interpolants. The basic technique is described in
[this paper](https://arxiv.org/abs/2004.04198). It currently supports
convolutional neural networks for image classification, using the
[Pytorch](https://pytorch.org/) framework.

Documentation on nnitp can be found
[here](https://nnitp.readthedocs.io/en/latest/)


# Install and Run

The pretrained model can be downloaded [here]()

Before install the program, run
'''pip install -r requirement.txt'''

To install the program, clone the repository and run
'''pip install .'''

To run the program
'''nnitp'''

Before you run the program, define environment varibale IMAGENETDIR and
PRETRAINEDDIR as the directory to the imagenet dataset and pretrianed model

'''export IMAGENETDIR='path/to/imagenet'''
'''export PRETRAINED='path/to/pretrained'''


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
