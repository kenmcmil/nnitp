#
# Copyright (c) Microsoft Corporation.
#

import sys
import os
import numpy as np
from importlib import import_module
from .error import Error
from .model_wrapper import compute_activation, compute_all_activation, sample_dataset
# Code for fetching models and datasets.
#
# TODO: this is dependent on torch framework.
#
# The models and datasets are defined by files in sub-directory
# `models` with names of the form `<name>_model.py` where `<name>` is
# the name to associated with the model/dataset. Each file is a python module
# containing two functions:
#
# - `get_data()` returns the datasets in the form `(x_train,y_train),(x_test,y_test)`
# - `get_model()` returns the trained model (usually stored in a file in the `models` directory)
#
# These functions are executed with `models` as working directory.
#



datasets = {}

#def import_models():
# This code scans the `models` directory and reads all of the modules
# into a dictionary `datasets`.

suffix = '_model.py'
model_path = [os.path.join(os.path.dirname(__file__),'models'),'.']
orig_sys_path = sys.path
sys.path.extend(model_path)
for dir in model_path:
    for fname in os.listdir(dir):
        if fname.endswith(suffix):
            modname = fname[0:-3]
            module = import_module(modname)
            name = fname[0:-len(suffix)]
            datasets[name] = module
sys.path = orig_sys_path

# Class `DataModel` is a combination of a dataset (training and test)
# and a trained model.

class DataModel(object):

    # Initial, the DataModel is unloaded, unless a name is given.

    def __init__(self, name = None):
        self.loaded = False
        self.load(name)

    # Load a `DataModel` by name.

    def load(self,name):
        self.name = name
        if name is not None:
            module = datasets[name]
            self.model = module.get_model()
            self.train_data,self.test_data = module.get_data()
            self.params = module.params if hasattr(module,'params') else {}
            self.loaded = True



    def set_sample_size(self,size:int):
        train_data = sample_dataset(self.train_data, size)
        test_data = sample_dataset(self.test_data, size)
        self._train_eval = ModelEval(self.model,train_data, self.name)
        self._test_eval = ModelEval(self.model,test_data, self.name)

    def output_layer(self) -> int:
        return len(self.model.layers) - 1




# Object for evaluating a model on an input set and caching the
# results. The constructor takes a model and some input data.  The
# `eval` method returns the activation value of layer `idx`. The method
# `set_pred` records a predicate `p` over layer `idx`. The method
# `split` returns a pair consisting of the the activations at layer `idx`
# when the predicate is true/false. Method `indices` returns a vector
# of the indices satisfying `p`.

class ModelEval(object):
    def __init__(self,model,data, name):
        self.model,self.data = model,data
        self.eval_cache = dict()
        self.name=name
    def eval(self,idx):
        if idx in self.eval_cache:
            return self.eval_cache[idx]
        print("evaluating layer {}".format(idx))

        res = compute_activation(self.model, idx, self.data, use_loader = True, name = self.name)

        print("done")
        self.eval_cache[idx] = res
        return res
    def set_pred(self,idx,p):
        self.split_cache = dict()
        self.cond = vect_eval(p,self.eval(idx))
    def set_layer_pred(self,lp):
        self.split_cache = dict()
        self.cond = lp.eval(self)
    def split(self,idx):
        if idx in self.split_cache:
            return self.split_cache[idx]
        def select(c):
            return np.compress(c,self.eval(idx),axis=0)
        res = (select(self.cond),select(np.logical_not(self.cond)))
        self.split_cache[idx] = res
        return res
    def indices(self):
        return np.compress(self.cond,np.arange(len(self.cond)))
    def eval_one(self,idx,input):
        data = input.reshape(1,*input.shape)
        return compute_activation(self.model,idx,data, name = self.name)[0]
    def eval_all(self,idx,data):
        return compute_activation(self.model,idx,data, name = self.name)
    def eval_all_layer(self):
        return compute_all_activation(self.model, self.data, use_loader = True, name = self.name)

#
# Evaluate a predicate on a vector.
#
# TODO: replace this with Predicate.map

def vect_eval(p,data):
    return np.array(list(map(p,data)))
