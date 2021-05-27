#
# Copyright (c) Microsoft Corporation.
#
import torch
import numpy as np
from model_mgr import DataModel,datasets
from itp import LayerPredicate, AndLayerPredicate, BoundPredicate
from itp import output_category_predicate
from typing import List,Optional,Tuple
from img import prepare_images
from bayesitp import Stats, interpolant, get_pred_cone, fraction, fractile
from copy import copy
import sys
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

import ruleex.deepred as dr


#
# Computation threads. We do computations in threads to avoid freezing
# the GUI.
#

# This thread is for loading the model.



def main(gamma,summary):
    name = "mnist"
    category = 0
    input_idx = 0
    size = 20000
    kwargs = {"alpha":0.95, "gamma": 0.6, "mu":0.9, "ensemble_size":1}
    kwargs["gamma"] = gamma
    data_model = DataModel()
    data_model.load(name)
    data_model.set_sample_size(size)
    train_eval = data_model._train_eval
    test_eval = data_model._test_eval
    layer_idxs = [56]
    avails = ['-1:input'] + ['{:0>2}'.format(i)+':'+ l
                            for i,l in enumerate(data_model.model.layers)]
    print(avails)
    layers = [avails[i] for i in layer_idxs if i>=0 and i <len(avails)]
    all_activation = train_eval.eval_all_layer()
    for i in range(len(all_activation)):
        all_activation[i]= all_activation[i].reshape(all_activation[i].shape[0],-1)
    params = dict()
    params["varbose"] =2
    params["build_first"] = True
    params["del_before_softmax"] = True
    rt = dr.deepred(all_activation, params)
    #rt_val = rt.eval_all(x)





    #F,N,P = stats.train_acc
    #train_prec = (N - F)/N if N != 0 else None
    #train_recall = (N - F)/P if P != 0 else None
    #F,N,P = stats.test_acc
    #test_prec = (N - F)/N if N != 0 else None
    #test_recall = (N - F)/P if P != 0 else None
    #complexity = len(itp.pred.args)








# Display the main window

if __name__ == '__main__':


    summary = {"train_prec":[],"test_prec":[],"train_recall":[], "test_recall":[], "complexity":[]}
    main(0.8, summary)



