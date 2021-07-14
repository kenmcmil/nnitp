#
# Copyright (c) Microsoft Corporation.
#

#
# Keras backend for nnitp
#

#from tensorflow.compat.v1.keras import backend as K
#from tensorflow.compat.v1.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import torch
import torch.nn as nn
from typing import Tuple
import itertools
import time
import numpy as np
from sklearn.utils import resample
from .utils import unflatten_unit
#from nnitp.models.models import Flatten
from nnitp.models.models import Flatten
import nnitp.models.resnet as resnet


# This class is the interface to torch models.
def sample_dataset(data, size, category = None):
    #size = min(size, len(data))
    #idx = range(size)
    targets = np.array(data.targets)
    size = min(size, len(targets))
    idx = []
    i1 = np.where(targets == category)[0]
    i2 = np.where(targets != category)[0]
    if len(i1) < size:
        i2 = resample(i2, n_samples= size-len(i1),replace =False, stratify = targets[i2])
        idx = np.concatenate((i1,i2))
    else:
        i1 = i1[:size//2]
        i2 = resample(i2, n_samples= size-len(i1),replace =False, stratify = targets[i2])
        idx = np.concatenate((i1,i2))
        #idx =range(len(data))
        #idx = resample(idx, n_samples= size,replace =False, stratify = targets[i2])
    data = torch.utils.data.Subset(data, idx)
    return data


def compute_activation(model,lidxs, test, use_loader = False, name =None):
    if use_loader:
        ret = dict()
        if name.startswith("imagenet"):
            data_loader = torch.utils.data.DataLoader(test, batch_size = 200, num_workers = 16, pin_memory = True)
        elif name.startswith("mnist"):
            data_loader = torch.utils.data.DataLoader(test, batch_size = 20000, num_workers = 16, pin_memory = True)
        else:
            data_loader = torch.utils.data.DataLoader(test, batch_size = 5000, num_workers = 16, pin_memory = True)
        for i, (inp, target) in enumerate(data_loader):
            s = time.time()
            temp = model.compute_activation(inp, lidxs)
            e = time.time()
            #print(i," :", e-s)
            for k in temp:
                if k in ret:
                    ret[k].append(temp[k])
                else:
                    ret[k] = [temp[k]]
            e1 = time.time()
            #print("total :", e1-s)
        for k in ret:
            ret[k] = torch.cat(ret[k])
        #ret = torch.cat(ret)
    else:
        ret = model.compute_activation(test, lidxs)
    for k in ret.keys():
        ret[k]=ret[k].numpy()
    return ret


# Computes the activation of layer `lidx` in model `model` over input
# data `test`. Layer index `-1` stands for the input data.
#


#def compute_activation(model,lidx,test, use_loader = False,  name = None):
#    if use_loader:
#        ret = []
#        if name.startswith("imagenet"):
#            data_loader = torch.utils.data.DataLoader(test, batch_size = 200, num_workers = 16, pin_memory = True)
#        else:
#            data_loader = torch.utils.data.DataLoader(test, batch_size = 5000, num_workers = 16, pin_memory = True)
#        for i, (inp, target) in enumerate(data_loader):
#            #s = time.time()
#            ret.append(model.compute_activation(lidx, inp).cpu())
#            #e = time.time()
#            #print(e-s)
#        ret = torch.cat(ret)
#    else:
#        ret = model.compute_activation(lidx, test)
#    return ret.cpu().numpy()

#get all layer and its name from model


def get_layers(model, layers, layers_name):
    for name,layer in model.named_children():
        if isinstance(layer, nn.Sequential) or isinstance(layer, resnet.BasicBlockM) or isinstance(layer, resnet.Bottleneck):
        #if isinstance(layer, nn.Sequential):
            get_layers(layer, layers,layers_name)
        if len(list(layer.children()))==0:
            layers.append(layer)
            layer_str = str(layer)
            end_idx= layer_str.find("(")
            layer_str = layer_str[:end_idx]
            layers_name.append(layer_str)




class Wrapper(object):

    # Constructor from a torch model.

    def __init__(self,model, inp_shape):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        #self._backend_session = K.get_session()
        self.inp_shape = inp_shape
        self.shape_hooks = []
        self.layer_shapes = []
        self._layers=[]
        self._layers_names=[]
        self._selected_sample = {}
        self.mid_out = dict()
        get_layers(self.model,self._layers,self._layers_names)
        self.layer_len = len(self._layers)

        #print(self._layers)
        #print(self._layers_names)

        self.init_shape()
        if isinstance(self.model, resnet.ResNetM):
            self.n,self.V,self.E = self.get_graph_resnet()
        else:
            self.n,self.V,self.E = self.get_graph_regular()

        #print(len(self.V))
        #print(len(self.E))
        #for v in self.V:
        #    print(v)
        #print(self.V[0])
        #print(self._layers)
        #print(len(self._layers))
        #print(len(self.layer_shapes))
        #print(self.layer_shapes)

    def init_shape(self):
        self.model.eval()
        for i,l in enumerate(self._layers):
            #self.fhooks.append(l.register_forward_hook(self.forward_hook(i)))
            self.shape_hooks.append(l.register_forward_hook(self.shape_hook(i)))
        x = torch.randn(self.inp_shape).to(self.device)
        with torch.no_grad():
          self.model(x)
        for hook in self.shape_hooks:
            hook.remove()

    def shape_hook(self,layer_idx):
        def hook(module,inp,out):
            self.layer_shapes.append(out.shape)
        return hook


    def forward_hook(self,layer_idx):
        #if str(layer_idx) in self._selected_sample:
        #    idxs = self._selected_sample[str(layer_idx)]
        #else:
        #    shape = self.layer_shapes[layer_idx]
        #    idxs = [slice(0,shape[0],1)]
        #    for i in shape[1:]:
        #        idxs.append(slice(0,i,2))
        #    self._selected_sample[str(layer_idx)] = idxs
        def hook(module,inp,out):
            self.mid_out[layer_idx] = out.detach().cpu()
        return hook

    def get_graph_regular(self):
        D = dict()
        n = 0
        V = []
        E = [[]]

        def record_hook(module, input, output):
            key = id(module)
            if key not in D:
                D[key] = len(V)
                V.append(module)

        hooks = []
        for module in self._layers:
            hooks.append(module.register_forward_hook(record_hook))
        x = torch.randn(self.inp_shape).to(self.device)
        y = self.model(x)
        for hook in hooks:
            hook.remove()

        n = len(V)
        E = [([False] * n) for i in range(n)]

        for i in range(n-1):
            E[i][i+1] = True

        return n, V, E


    def get_graph_resnet(self):
        D = dict()
        n = 0
        V = []
        E = [[]]

        def record_hook(module, input, output):
            key = id(module)
            if key not in D:
                D[key] = len(V)
                V.append(module)

        def add_edge(src, dst):
            i = D[id(src)]
            j = D[id(dst)]
            E[i][j] = True

        def add_chain(ls):
            for i in range(len(ls) - 1):
                add_edge(ls[i], ls[i + 1])

        hooks = []
        for module in self._layers:
            hooks.append(module.register_forward_hook(record_hook))
        x = torch.randn(self.inp_shape).to(self.device)
        y = self.model(x)
        for hook in hooks:
            hook.remove()

        n = len(V)
        E = [([False] * n) for i in range(n)]

        chain = [self.model.conv1, self.model.bn1, self.model.relu1]
        if hasattr(self.model, "maxpool"):
            chain.append(self.model.maxpool)
        add_chain(chain)

        src = [chain[-1]]
        for module in self.model.modules():
            if isinstance(module, resnet.BasicBlockM):
                chain = [module.conv1, module.bn1, module.relu1,
                         module.conv2, module.bn2, module.relu2]
                add_chain(chain)
                dst = [module.conv1]
                src_ = [module.relu2]

                if module.downsample is not None:
                    chain = list(module.downsample.children())
                    add_chain(chain)
                    dst.append(chain[0])
                    add_edge(chain[-1], module.relu2)
                else:
                    dst.append(module.relu2)

                for s in src:
                    for d in dst:
                        add_edge(s, d)
                src = src_
                dst = []

            if isinstance(module, resnet.Bottleneck):
                chain = [module.conv1, module.bn1, module.relu1,
                         module.conv2, module.bn2, module.relu2,
                         module.conv3, module.bn3, module.relu3]
                add_chain(chain)
                dst = [module.conv1]
                src_ = [module.relu3]

                if module.downsample is not None:
                    chain = list(module.downsample.children())
                    add_chain(chain)
                    dst.append(chain[0])
                    add_edge(chain[-1], module.relu3)
                else:
                    dst.append(module.relu3)

                for s in src:
                    for d in dst:
                        add_edge(s, d)
                src = src_
                dst = []
        chain = [self.model.avgpool, self.model.flatten, self.model.fc]
        for s in src:
            add_edge(s, chain[0])
        add_chain(chain)

        return n, V, E




    # To use this model in given model in a thread, we have to set it
    # up as the default Keras session and also set up the tensorflow
    # default graph. This method returns a context object suitable for
    # this purpose.  To run inference using model `foo`, you have to
    # use `with session(): ...`. TODO: Really need to expose this?

    #def session(self):
    #    K.set_session(self._backend_session)
    #    return self._backend_session.graph.as_default()

    # Use tensorflow to compute the activation of layer `lidx` in
    # model `model` over input data `test`. Layer index `-1` stands
    # for the input data.
    #

    def compute_activation(self, test, lidxs):
        self.model.eval()
        if isinstance(lidxs, int):
            lidxs = set([lidxs])
        if not torch.is_tensor(test):
            test = torch.tensor(test)
        test = test.to(self.device)
        self.fhook = []
        ret = dict()
        for i in lidxs:
            if i < 0:
                ret[i] = test.detach().cpu()
            else:
                l = self.get_layer(i)
                self.fhook.append(l.register_forward_hook(self.forward_hook(i)))
        with torch.no_grad():
            self.model(test)
        for hook in self.fhook:
            hook.remove()

        ret.update(self.mid_out)
        self.mid_out = dict()
        return ret

    #def get_weight(self):
    #    self.inp_weight = dict()
    #    self.out_weight = dict()
    #    for i in range(self.n):
    #        layer = self.get_layer(i)
    #        out_shape = self.layer_shape(i)
    #        if i == 0:
    #            prev = [-1]
    #            inp_shape = self.layer_shape(-1)
    #        else:
    #            prev = []
    #            for j in range(i):
    #                if self.E[j][i]:
    #                    prev.append(j)
    #            inp_shape = self.layer_shape(prev[-1])
    #        if isinstance(layer, nn.Conv2d):
    #            x = torch.randn([1]+inp_shape).to(self.device)
    #            with torch.no_grad():
    #                w = layer(x)
    #            self.inp_weight[i] = w
    #            for k in self.prev:
    #                self.out_weight[k] = 







    #def compute_importance(self, data, name):
    #    if name.startswith("imagenet"):
    #        data_loader = torch.utils.data.DataLoader(data, batch_size = 200, num_workers = 16, pin_memory = True)
    #    elif name.startswith("mnist"):
    #        data_loader = torch.utils.data.DataLoader(data, batch_size = 20000, num_workers = 16, pin_memory = True)
    #    else:
    #        data_loader = torch.utils.data.DataLoader(data, batch_size = 5000, num_workers = 16, pin_memory = True)
    #    import PyIFS
    #    inf = PyIFS.InfFS()
    #    x = []
    #    for i, (inp,out) in enumerate(data_loader):
    #        x.append(self.model(inp.to(self.device)).detach().cpu())
    #    x = torch.cat(x).numpy()
    #    y = data.targets
    #    [ranked,weight] = inf.infFS(x, y, 0.5, 0, 0)
    #    weight = torch.tensor(weight).float().to(self.device)
    #    self.importance = [weight]
    #    cur = weight
    #    for layer in reversed(self._layers):
    #        if hasattr(layer, "weight"):
    #            print(cur.shape)
    #            print(layer.weight.data.shape)
    #            cur = torch.matmul(layer.weight.data.T, cur)
    #        self.importance = [cur] + self.importance
    #    self.importance = self.importance[1:]






    #def compute_activation(self,lidx,test):
    #    self.model.eval()
    #    if not torch.is_tensor(test):
    #        test = torch.tensor(test)
    #    test = test.to(self.device)
    #    if lidx < 0:
    #        return test
    #    self.mid_out = []
    #    l = self.get_layer(lidx)
    #    self.fhook = l.register_forward_hook(self.forward_hook(lidx))
    #    with torch.no_grad():
    #      self.model(test)
    #    self.fhook.remove()

    #    return self.mid_out[0]


    # Get the shape of a given layer's tensor, or the input shape if
    # layer is -1.

    def layer_shape(self,layer):
        if layer == -1:
            return self.inp_shape
        else:
            return self.layer_shapes[layer]
        #return self.model.input_shape if layer == -1 else self.model.layers[layer].output_shape

    # return specific layer
    def get_layer(self,idx):
        return self._layers[idx]


    # Return  list of layer names.
    @property
    def layers(self):
        return self._layers_names
    #[layer.name for layer in self.model.layers]

    # Get the slice at layer `n` (-1 for input) that is relevant to a
    # slice `slc` at layer `n1`. TODO: doesn't really belong here since it
    # is toolkit-dependent. TODO: Not sure if this gives correct result
    # for even convolutional kernel sizes in case of padding == 'same', as
    # Keras docs don't say how padding is done in this case (i.e., whether
    # larger padding is used on left/bottom or right/top).

    # TODO: change the `slc` argument to a list of python slice objects.

    #def get_cone(self,n,n1,slc) -> Tuple:
    #    #print("----------start-----------")
    #    model = self.model
    #    model.eval()
    #    while n1 > n:
    #        layer = self.get_layer(n1)
    #        layer_inp = self.layer_shape(n1-1)
    #        layer_out = self.layer_shape(n1)
    #        if isinstance(layer,nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
    #            c_in = layer_inp[1]
    #            H = layer_inp[2]
    #            W = layer_inp[3]
    #            padding = layer.padding
    #            dilation = layer.dilation
    #            stride = layer.stride
    #            kernel = layer.kernel_size
    #            if isinstance(padding,int):
    #                padding = tuple([padding, padding])
    #            if isinstance(kernel,int):
    #                kernel = tuple([kernel, kernel])
    #            if isinstance(dilation,int):
    #                dilation = tuple([dilation, dilation])
    #            if isinstance(stride,int):
    #                stride = tuple([stride, stride])


    #            H_min = max(0, slc[0][1]*stride[0]-padding[0])
    #            W_min = max(0,slc[0][2]*stride[1]-padding[1])
    #            H_max = min(H-1, slc[1][1]*stride[0]+dilation[0]*(kernel[0]-1)-padding[0])
    #            W_max = min(W-1, slc[1][2]*stride[1]+dilation[1]*(kernel[1]-1)-padding[1])

    #            if isinstance(layer, nn.MaxPool2d):
    #                slc = ((slc[0][0], H_min, W_min),
    #                        (slc[1][0],H_max,W_max))
    #            else:
    #                slc = ((0, H_min, W_min),
    #                        (c_in-1,H_max,W_max))

    #        elif isinstance(layer,Flatten):
    #            shape = layer_inp[1:]
    #            slc = (unflatten_unit(shape,slc[0]),unflatten_unit(shape,slc[1]))
    #        elif isinstance(layer,nn.Linear):
    #            shape = layer_inp[1:]
    #            slc = (tuple(0 for x in shape),tuple(x-1 for x in shape))
    #        elif layer_inp == layer_out:
    #            pass
    #        else:
    #            print ("Cannot compute dependency cone for layer of type {}.".format(type(layer)))
    #            exit(1)
    #        n1 -= 1
    #    return tuple(slice(x,y+1) for x,y in zip(slc[0],slc[1]))




    def get_cone(self,n,n1,cone) -> Tuple:
        model = self.model
        model.eval()
        cone_lists = [set() for i in range(n1-n+1)]
        cone_lists[-1].update(cone)
        while n1 > n:
            layer = self.get_layer(n1)
            layer_out = self.layer_shape(n1)
            inp = True
            for i in range(self.n):
                if self.E[i][n1]:
                    inp = False
                    layer_inp = self.layer_shape(i)
                    break
            if inp:
                layer_inp = self.inp_shape


            cones = cone_lists[n1-n]
            if layer_inp == layer_out:
                for i in range(n1):
                    if self.E[i][n1]:
                        cone_lists[i-n].update(cones)
            else:
                for cone in cones:
                    if isinstance(layer,nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                        C_in = layer_inp[1]
                        H = layer_inp[2]
                        W = layer_inp[3]
                        padding = layer.padding
                        dilation = layer.dilation
                        stride = layer.stride
                        kernel = layer.kernel_size
                        if isinstance(padding,int):
                            padding = tuple([padding, padding])
                        if isinstance(kernel,int):
                            kernel = tuple([kernel, kernel])
                        if isinstance(dilation,int):
                            dilation = tuple([dilation, dilation])
                        if isinstance(stride,int):
                            stride = tuple([stride, stride])


                        H_min = max(0, cone[1]*stride[0]-padding[0])
                        W_min = max(0,cone[2]*stride[1]-padding[1])
                        H_max = min(H-1, cone[1]*stride[0]+dilation[0]*(kernel[0]-1)-padding[0])
                        W_max = min(W-1, cone[2]*stride[1]+dilation[1]*(kernel[1]-1)-padding[1])

                        if isinstance(layer, nn.MaxPool2d):
                            C_min = cone[0]
                            C_max = cone[0]
                        else:
                            C_min = 0
                            C_max = C_in - 1
                        new_cones = [range(C_min,C_max+1), range(H_min, H_max+1), range(W_min, W_max+1)]

                        new_cones = set(itertools.product(*new_cones))
                    #Not accurate, the implementation of adaptiveavgpool is unclear
                    elif isinstance(layer, nn.AdaptiveAvgPool2d):
                        c_in = layer_inp[1]
                        H_in = layer_inp[2]
                        W_in = layer_inp[3]
                        H_out = layer_out[2]
                        W_out = layer_out[3]
                        padding = (0,0)
                        stride = [H_in//H_out, W_in//W_out]
                        kernel = [H_in - (H_out-1)*stride[0], W_in - (W_out-1)*stride[1]]


                        H_min = max(0, cone[1]*stride[0]-padding[0])
                        W_min = max(0, cone[2]*stride[1]-padding[1])
                        H_max = min(H_in-1, cone[1]*stride[0]+(kernel[0]-1)-padding[0])
                        W_max = min(W_in-1, cone[2]*stride[1]+(kernel[1]-1)-padding[1])

                        C_min = cone[0]
                        C_max = cone[0]
                        new_cones = [range(C_min,C_max+1), range(H_min, H_max+1), range(W_min, W_max+1)]

                        new_cones = set(itertools.product(*new_cones))

                    elif isinstance(layer, nn.AvgPool2d):
                        c_in = layer_inp[1]
                        H = layer_inp[2]
                        W = layer_inp[3]
                        padding = layer.padding
                        stride = layer.stride
                        kernel = layer.kernel_size
                        if isinstance(padding,int):
                            padding = tuple([padding, padding])
                        if isinstance(kernel,int):
                            kernel = tuple([kernel, kernel])
                        if isinstance(stride,int):
                            stride = tuple([stride, stride])


                        H_min = max(0, cone[1]*stride[0]-padding[0])
                        W_min = max(0, cone[2]*stride[1]-padding[1])
                        H_max = min(H-1, cone[1]*stride[0]+(kernel[0]-1)-padding[0])
                        W_max = min(W-1, cone[2]*stride[1]+(kernel[1]-1)-padding[1])

                        C_min = cone[0]
                        C_max = cone[0]
                        new_cones = [range(C_min,C_max+1), range(H_min, H_max+1), range(W_min, W_max+1)]

                        new_cones = set(itertools.product(*new_cones))

                    elif isinstance(layer,Flatten):
                        shape = layer_inp[1:]
                        new_cones = {unflatten_unit(shape,cone)}

                    elif isinstance(layer,nn.Linear):
                        shape = layer_inp[1:]
                        #new_slc = (tuple(0 for x in shape),tuple(x-1 for x in shape))
                        new_cones = []
                        for x in shape:
                            new_cones.append(range(x))
                        new_cones = set(itertools.product(*new_cones))
                    else:
                        print ("Cannot compute dependency cone for layer of type {}.".format(type(layer)))
                        exit(1)
                    #layer_id = id(layer)
                    #if new_cones == set():
                    #    print("equal")
                    #print(new_cones)
                    inp = True
                    for i in range(self.n):
                        if self.E[i][n1]:
                            inp = False
                            cone_lists[i-n].update(new_cones)
                    if inp:
                        cone_lists[0].update(new_cones)


            n1 -= 1
        cones = cone_lists[0]
        return cones
        #return tuple(slice(x,y+1) for x,y in zip(slc[0],slc[1]))





