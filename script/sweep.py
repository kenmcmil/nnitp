#
# Copyright (c) Microsoft Corporation.
#
import torch
import numpy as np
from typing import List,Optional,Tuple
from copy import copy
import sys
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes
import os
import time
import gc
import argparse
import random
from nnitp.model_mgr import DataModel,datasets
from nnitp.itp import LayerPredicate, AndLayerPredicate, BoundPredicate,And
from nnitp.itp import output_category_predicate
from nnitp.img import prepare_images
from nnitp.bayesitp import Stats, interpolant, get_pred_cone, fraction, fractile, check_itp
#
# Computation threads. We do computations in threads to avoid freezing
# the GUI.
#

# This thread is for loading the model.



def main(param,summary):
    name = param["name"]
    size = param["size"]
    category = param["category"]
    input_idx = param["input_idx"]
    if name.startswith("imagenet"):
        kwargs = {"alpha":0.85, "ensemble_size":1}
    elif name.startswith("cifar10"):
        kwargs = {"alpha":0.95, "ensemble_size":1}
    elif name.startswith("mnist"):
        kwargs = {"alpha":0.98, "ensemble_size":1}
    kwargs["mu"] = param["mu"]
    kwargs["gamma"] = param["gamma"]

    s1 = time.time()
    data_model = DataModel()
    data_model.load(name)
    data_model.set_sample_size(size, category = category)
    train_eval = data_model._train_eval
    test_eval = data_model._test_eval
    layer_idxs = [param["layer"]]
    avails = ['-1:input'] + ['{:0>2}'.format(i)+':'+ l
                            for i,l in enumerate(data_model.model.layers)]
    last = len(avails)-2
    train_eval.set_idxs(layer_idxs + [-1, last])
    test_eval.set_idxs(layer_idxs + [last])
    print(avails)
    layers = [avails[i+1] for i in layer_idxs if i+1>=0 and i+1 <len(avails)]
    s2 = time.time()
    conc = output_category_predicate(data_model, category)
    s3 = time.time()
    compset = conc.sat(train_eval)
    inp = compset[input_idx]
    s4 = time.time()


    def previous_layer(layers, cur):
        cur_layers = [int(x.split(':')[0]) for x in layers]
        cur_layers = [x for x in sorted(cur_layers) if x < cur]
        return cur_layers[-1] if cur_layers else -1


    inp = inp.reshape(1,*inp.shape)
    layer = previous_layer(layers, conc.layer)
    #itp_i,stats = interpolant(data_model,layer,
    #                                     inp,conc,**kwargs)
    itp,stats = interpolant(data_model,layer,
                                         inp,conc,**kwargs)
    F,N,P = stats.train_acc
    train_prec = (N - F)/N if N != 0 else None
    train_recall = (N - F)/P if P != 0 else None
    print("train_prec: ", train_prec)
    print("train_recall: ", train_recall)
    F,N,P = stats.test_acc
    test_prec = (N - F)/N if N != 0 else None
    test_recall = (N - F)/P if P != 0 else None
    complexity = len(itp.pred.args)
    print("test_prec: ", test_prec)
    print("test_recall: ", test_recall)
    print("complexity: ", complexity)





    #itps = []
    #for pred in itp_i.pred.args:
    #    itps.append(LayerPredicate(itp_i.layer,pred))

    #print("-----------------------------------------------")
    #F,N,P = stats.train_acc
    #print("training N: ", N)
    #N_list = stats.train_ensemble_acc
    #for i, r in enumerate(N_list):
    #    print("n",i, " :",r[1])


    #F,N,P = stats.test_acc
    #print("testing N: ", N)
    #N_list = stats.test_ensemble_acc
    #for i, r in enumerate(N_list):
    #    print("n",i, " :",r[1])
    #print("-----------------------------------------------")




    #train_eval.remove_cache(conc.layer)
    #test_eval.remove_cache(conc.layer)
    #new_itps = []
    #for cur in itps:
    #    itp,stats = interpolant(data_model,2,
    #                                        inp,cur,**kwargs)
    #    print(itp.pred)
    #    new_itps.append(itp.pred)
    #    F,N,P = stats.train_acc
    #    train_prec = (N - F)/N if N != 0 else None
    #    train_recall = (N - F)/P if P != 0 else None
    #    print("train_prec: ", train_prec)
    #    print("train_recall: ", train_recall)
    #    F,N,P = stats.test_acc
    #    test_prec = (N - F)/N if N != 0 else None
    #    test_recall = (N - F)/P if P != 0 else None
    #    complexity = len(itp.pred.args)
    #    print("test_prec: ", test_prec)
    #    print("test_recall: ", test_recall)
    #    print("complexity: ", complexity)
    #l_pred = And(*new_itps)
    #print(l_pred)
    #F,N,P = check_itp(train_eval, 2, l_pred ,itp_i.layer, itp_i.pred)
    #print("F: ", F)
    #print("N: ", N)
    #print("P: ", P)
    #train_prec = (N - F)/N if N != 0 else None
    #train_recall = (N - F)/P if P != 0 else None
    #print("train_prec: ", train_prec)
    #print("train_recall: ", train_recall)
    #F,N,P = check_itp(test_eval, 2, l_pred,itp_i.layer, itp_i.pred)
    #print("F: ", F)
    #print("N: ", N)
    #print("P: ", P)
    #test_prec = (N - F)/N if N != 0 else None
    #test_recall = (N - F)/P if P != 0 else None
    #complexity = len(itp.pred.args)
    #print("test_prec: ", test_prec)
    #print("test_recall: ", test_recall)
    #print("complexity: ", complexity)


    if train_prec is not None and test_prec is not None and train_recall is not None and test_recall is not None and complexity is not None:
        summary["train_prec"].append(train_prec)
        summary["test_prec"].append(test_prec)
        summary["train_recall"].append(train_recall)
        summary["test_recall"].append(test_recall)
        summary["complexity"].append(complexity)

    #e = time.time()
    #print("preparation time: ", s2-s1)
    #print("output predicate time: ", s3-s2)
    #print("sat computation time: ", s4-s3)
    #print("interpolant time: ", s5-s4)
    #print("total time: ", e-s1)


def plot(logs,mu, save_dir):
    lines = []
    x = logs["gamma"]
    yprec = logs["test_prec"]
    yrec = logs["test_recall"]
    ysize = logs["complexity"]
    lines.append({'key':'precision, mu={:2.1}'.format(mu), 'x':x, 'y':yprec,'axis':0})
    lines.append({'key':'recall, mu={:2.1}'.format(mu), 'x':x, 'y':yrec,'axis':1})
    lines.append({'key':'complexity, mu={:2.1}'.format(mu), 'x':x, 'y':ysize,'axis':2})

    fig = plt.figure(figsize=(7,4.8))
    print(plt.rcParams.get('figure.figsize'))
    plt.rcParams.update({'font.size': 12})

    host = fig.add_axes([0.15, 0.1, 0.65, 0.8], axes_class=HostAxes)
    par1 = ParasiteAxes(host, sharex=host)
    par2 = ParasiteAxes(host, sharex=host)
    host.parasites.append(par1)
    host.parasites.append(par2)


    host.axis["right"].set_visible(False)
    par1.axis["right"].set_visible(True)
    par1.axis["right"].major_ticklabels.set_visible(True)
    par1.axis["right"].label.set_visible(True)


    offset = (60, 0)
    par2.axis["right2"] = par2.new_fixed_axis(loc="right", offset=offset)

    fig.add_axes(host)


    host.set_xlabel("gamma")
    host.set_ylabel("Precision")
    par1.set_ylabel("Recall")
    par2.set_ylabel("Complexity")

    axes = [host,par1,par2]
    things = []
    for idx,line in enumerate(lines):
        mkr = ['|','x','o'][idx]
        p, = axes[line['axis']].plot(line['x'],line['y'],marker=mkr,label=line['key'])
        things.append(p)


    host.legend(loc="center right",fontsize="x-small")

    host.axis["left"].label.set_color(things[0].get_color())
    par1.axis["right"].label.set_color(things[1].get_color())
    par2.axis["right2"].label.set_color(things[2].get_color())

    plt.savefig(save_dir,dpi=fig.dpi, bbox_inches = 'tight')






# Display the main window

if __name__ == '__main__':
    parser = argparse.ArgumentParser("sweep")
    parser.add_argument('--experiment', type=str, default='mnist', choices=['cifar10','imagenet_vgg19', "imagenet_resnet34", 'mnist','cifar10_resnet34'],help='experiment to run')
    parser.add_argument('--gamma_min', type=float, default=0.45, help='minimum value of gamma for sweeping')
    parser.add_argument('--gamma_max', type=float, default=0.8, help='maximum value of gamma for sweeping')
    parser.add_argument('--gamma_step', type=float, default=0.05, help='step of gamma of sweeping')
    parser.add_argument('--mu_min', type=float, default=0.7, help='minimum value of gamma for sweeping')
    parser.add_argument('--mu_max', type=float, default=0.9, help='maximum value of gamma for sweeping')
    parser.add_argument('--mu_step', type=float, default=0.2, help='step of mu of sweeping')
    parser.add_argument('--num_images', type=int, default=5, help='num of images to average')
    parser.add_argument('--layer', type=int, default=2, help='layer selected for running experiment')
    parser.add_argument('--sample_size', type=int, default=20000, help='num of sample size')
    parser.add_argument('--all_category', action = "store_true", help='whether to use all category')
    parser.add_argument('--category', type=int, default=0, help='index of selected category')
    parser.add_argument('--save_dir', type=str, default='./runs', help='path to save the result')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    path = "{}/{}_{}_{}".format(args.save_dir,args.experiment, args.layer, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(path):
        os.mkdir(path)

    gamma_range = np.arange(args.gamma_min,args.gamma_max+0.1*args.gamma_step,args.gamma_step)
    mu_range = np.arange(args.mu_min,args.mu_max+0.1*args.mu_step,args.mu_step)

    categories = {"cifar10":10, "imagenet_vgg19":1000, "mnist":10}

    param = {}
    param["name"] = args.experiment
    param["layer"]= args.layer
    param["size"]= args.sample_size
    for mu in mu_range:
        param["mu"]= mu
        logs = {"gamma":[],"train_prec":[],"test_prec":[],"train_recall":[], "test_recall":[], "complexity":[]}
        for gamma in gamma_range:

            param["gamma"] = gamma
            summary = {"train_prec":[],"test_prec":[],"train_recall":[], "test_recall":[], "complexity":[]}
            if args.all_category:
                cats = range(categories[param["name"]])
            else:
                cats = [args.category]
            for cat in cats:
                param["category"] = cat
                for j in range(args.num_images):
                    param["input_idx"]=j
                    main(param, summary)

            logs["gamma"].append(gamma)
            logs["train_prec"].append(mean(summary["train_prec"]))
            logs["test_prec"].append(mean(summary["test_prec"]))
            logs["train_recall"].append(mean(summary["train_recall"]))
            logs["test_recall"].append(mean(summary["test_recall"]))
            logs["complexity"].append(mean(summary["complexity"]))
            data_save = "%s/data_%2.1f.pth"%(path, mu)
            torch.save(logs, data_save)

        data_save = "%s/data_%2.1f.pth"%(path, mu)
        fig_save = "%s/sweep_%2.1f.png"%(path, mu)
        torch.save(logs, data_save)
        plot(logs,mu, fig_save)



