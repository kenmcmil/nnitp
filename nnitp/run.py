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



#
# Computation threads. We do computations in threads to avoid freezing
# the GUI.
#

# This thread is for loading the model.



def main(param,summary, layer):
    name = "cifar10_vgg19"
    category = 0
    input_idx = 0
    size = 20000
    kwargs = {"alpha":0.95, "gamma": 0.6, "mu":0.9, "ensemble_size":1}
    kwargs.update(param)
    #kwargs["gamma"] = gamma
    data_model = DataModel()
    data_model.load(name)
    data_model.set_sample_size(size)
    train_eval = data_model._train_eval
    test_eval = data_model._test_eval
    layer_idxs = [layer]
    avails = ['-1:input'] + ['{:0>2}'.format(i)+':'+ l
                            for i,l in enumerate(data_model.model.layers)]
    print(avails)
    layers = [avails[i] for i in layer_idxs if i>=0 and i <len(avails)]

    conc = output_category_predicate(data_model, category)
    compset = conc.sat(train_eval)
    inp = compset[input_idx]

    def previous_layer(layers, cur):
        cur_layers = [int(x.split(':')[0]) for x in layers]
        cur_layers = [x for x in sorted(cur_layers) if x < cur]
        return cur_layers[-1] if cur_layers else -1


    inp = inp.reshape(1,*inp.shape)
    layer = previous_layer(layers, conc.layer)
    itp,stats = interpolant(data_model,layer,
                                         inp,conc,**kwargs)

    print(itp)
    cones = []

    for conj in itp.conjs():
        cone = get_pred_cone(data_model.model, conj)
        #cone = data_model.pred_cone(conj)
        cones.append(cone)
    print(cones)
    pixels = []
    y58
    for sidx, slc in enumerate(cones):
        y = abs(slc[1].stop - slc[1].start)
        x = abs(slc[2].stop - slc[2].start)
        pixel = x*y
        pixels.append(pixel)
    print(pixels)
    print(sum(pixels))


    F,N,P = stats.train_acc
    train_prec = (N - F)/N if N != 0 else None
    train_recall = (N - F)/P if P != 0 else None
    F,N,P = stats.test_acc
    test_prec = (N - F)/N if N != 0 else None
    test_recall = (N - F)/P if P != 0 else None
    complexity = len(itp.pred.args)
    #rint(complexity)
    if train_prec and test_prec and train_recall and test_recall and complexity:
        summary["train_prec"].append(train_prec)
        summary["test_prec"].append(test_prec)
        summary["train_recall"].append(train_recall)
        summary["test_recall"].append(test_recall)
        summary["complexity"].append(complexity)


    #print("Interpolant: {}\n".format(itp))
    #print("stats: {}\n".format(stats))


def plot(logs,mu):
    lines = []
    x = logs["gamma"]
    yprec = logs["train_prec"]
    yrec = logs["train_recall"]
    ysize = logs["complexity"]
    lines.append({'key':'precision, mu={}'.format(mu), 'x':x, 'y':yprec,'axis':0})
    lines.append({'key':'recall, mu={}'.format(mu), 'x':x, 'y':yrec,'axis':1})
    lines.append({'key':'complexity, mu={}'.format(mu), 'x':x, 'y':ysize,'axis':2})

    fig = plt.figure(figsize=(7,4.8))
    print(plt.rcParams.get('figure.figsize'))
    plt.rcParams.update({'font.size': 12})

    host = fig.add_axes([0.15, 0.1, 0.65, 0.8], axes_class=HostAxes)
    #host = HostAxes(fig, [0.15, 0.1, 0.60, 0.8])
    par1 = ParasiteAxes(host, sharex=host)
    par2 = ParasiteAxes(host, sharex=host)
    host.parasites.append(par1)
    host.parasites.append(par2)

    #host.set_ylabel("Precision")
    #host.set_xlabel("gamma")

    host.axis["right"].set_visible(False)
    par1.axis["right"].set_visible(True)
    par1.axis["right"].major_ticklabels.set_visible(True)
    par1.axis["right"].label.set_visible(True)
    #par1.set_ylabel("Recall")


    #par2.set_ylabel("Complexity")
    offset = (60, 0)
    #new_axisline = par2.get_grid_helper().new_fixed_axis
    #par2.axis["right2"] = new_axisline(loc="right", axes=par2, offset=offset)
    par2.axis["right2"] = par2.new_fixed_axis(loc="right", offset=offset)

    fig.add_axes(host)


    host.set_xlabel("gamma")
    host.set_ylabel("Precision")
    par1.set_ylabel("Recall")
    par2.set_ylabel("Complexity")

    axes = [host,par1,par2]
    things = []
    for idx,line in enumerate(lines):
        print(line)
        mkr = ['|','x','o'][idx]
        p, = axes[line['axis']].plot(line['x'],line['y'],marker=mkr,label=line['key'])
        things.append(p)
    #plt.show()

    # p1, = host.plot([0, 1, 2], [0, 1, 2], label="Precision")
    # p2, = par1.plot([0, 1, 2], [0, 3, 2], label="Recall")
    # p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Complexity")

#    par1.set_ylim(0, 4)
#    par2.set_ylim(1, 65)

    host.legend(loc="upper left",fontsize="x-small")

    host.axis["left"].label.set_color(things[0].get_color())
    par1.axis["right"].label.set_color(things[1].get_color())
    par2.axis["right2"].label.set_color(things[2].get_color())
    #plt.show()

    plt.savefig("vgg_6/sweep0.7.png",dpi=fig.dpi, bbox_inches = 'tight')
    plt.show()






# Display the main window

if __name__ == '__main__':
    #np.random.seed(123)
    #torch.manual_seed(123)
    #torch.cuda.manual_seed(123)

    summary = {"train_prec":[],"test_prec":[],"train_recall":[], "test_recall":[], "complexity":[]}
    param = {"mu":0.9, "gamma":0.75}
    main(param, summary, layer = 13)

    #param = {"mu":0.9, "gamma":0.8}
    #logs = {"gamma":[],"train_prec":[],"test_prec":[],"train_recall":[], "test_recall":[], "complexity":[]}
    #for i in range(8):
    #    summary = {"train_prec":[],"test_prec":[],"train_recall":[], "test_recall":[], "complexity":[]}
    #    for j in range(5):
    #      param["gamma"] = 0.45+i*0.05
    #      main(param, summary,13)
    #    logs["gamma"].append(0.45+i*0.05)
    #    logs["train_prec"].append(mean(summary["train_prec"]))
    #    logs["test_prec"].append(mean(summary["test_prec"]))
    #    logs["train_recall"].append(mean(summary["train_recall"]))
    #    logs["test_recall"].append(mean(summary["test_recall"]))
    #    logs["complexity"].append(mean(summary["complexity"]))
    #torch.save(logs, "vgg_13/vgg_data0.9.pth")
    #plot(logs,0.9)



