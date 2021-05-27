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
from utils import unflatten_unit
#from nnitp.models.models import Flatten
from models.models import Flatten

# This class is the interface to torch models.
def sample_dataset(data, size):
    size = min(len(data), size)
    data = torch.utils.data.Subset(data, range(size))
    return data


def compute_all_activation(model,test, use_loader = False):
    if use_loader:
        temp = []
        data_loader = torch.utils.data.DataLoader(test, batch_size = 5000, num_workers = 4, pin_memory = True)
        for i, (inp, target) in enumerate(data_loader):
            temp.append(model.compute_all_activation(inp))
        ret=[]
        for i in range(len(temp[0])):
            cur = []
            for j in range(len(temp)):
                cur.append(temp[j][i])
            ret.append(torch.cat(cur))
        #ret = torch.cat(ret)
    else:
        ret = model.compute_all_activation(test)

    for i in range(len(ret)):
        ret[i]=ret[i].cpu().numpy()
    return ret


# Computes the activation of layer `lidx` in model `model` over input
# data `test`. Layer index `-1` stands for the input data.
#

def compute_activation(model,lidx,test, use_loader = False, all_layer = False):
    if use_loader:
        ret = []
        data_loader = torch.utils.data.DataLoader(test, batch_size = 5000, num_workers = 4, pin_memory = True)
        for i, (inp, target) in enumerate(data_loader):
            ret.append(model.compute_activation(lidx, inp, all_layer = all_layer).cpu())
        ret = torch.cat(ret)
    else:
        ret = model.compute_activation(lidx, test, all_layer = all_layer)
    return ret.cpu().numpy()

#get all layer and its name from model


def get_layers(model, layers, layers_name):
    for name,layer in model.named_children():
        if isinstance(layer, nn.Sequential):
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
        #self.mid_out = []
        #self.fhooks = []
        self.shape_hooks = []
        self.layer_shapes = []
        self._layers=[]
        self._layers_names=[]
        get_layers(self.model,self._layers,self._layers_names)
        self.layer_len = len(self._layers)

        self.init_shape()
        self.n,self.V,self.E = self.get_graph_resnet()
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
        def hook(module,inp,out):
            self.mid_out.append(out.detach())
        return hook

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
        y = self.model(y)
        for hook in hooks:
            hook.remove()

        n = len(V)
        E = [([False] * n) for i in range(n)]

        chain = [self.model.conv1, self.model.bn1, self.model.relu1]
        add_chain(chain)

        src = [self.model.relu1]
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

    def compute_all_activation(self,test):
        self.model.eval()
        if not torch.is_tensor(test):
            test = torch.tensor(test)
        test = test.to(self.device)
        self.mid_out = []
        self.fhook = []
        for i, l in enumerate(self._layers):
            self.fhook.append(l.register_forward_hook(self.forward_hook(i)))
        with torch.no_grad():
            out = self.model(test)
        for hook in self.fhook:
            hook.remove()

        ret = [test.cpu()]+self.mid_out+[out.cpu()]

        return ret

    def compute_activation(self,lidx,test):
        self.model.eval()
        if not torch.is_tensor(test):
            test = torch.tensor(test)
        test = test.to(self.device)
        if lidx < 0:
            return test
        self.mid_out = []
        l = self.get_layer(lidx)
        self.fhook = l.register_forward_hook(self.forward_hook(lidx))
        with torch.no_grad():
          self.model(test)
        self.fhook.remove()

        return self.mid_out[0]


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

    def get_cone(self,n,n1,slc) -> Tuple:
        #print("----------start-----------")
        model = self.model
        model.eval()
        while n1 > n:
            layer = self.get_layer(n1)
            layer_inp = self.layer_shape(n1-1)
            layer_out = self.layer_shape(n1)
            if isinstance(layer,nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                c_in = layer_inp[1]
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


                H_min = max(0, slc[0][1]*stride[0]-padding[0])
                W_min = max(0,slc[0][2]*stride[1]-padding[1])
                H_max = min(H-1, slc[1][1]*stride[0]+dilation[0]*(kernel[0]-1)-padding[0])
                W_max = min(W-1, slc[1][2]*stride[1]+dilation[1]*(kernel[1]-1)-padding[1])

                if isinstance(layer, nn.MaxPool2d):
                    slc = ((slc[0][0], H_min, W_min),
                            (slc[1][0],H_max,W_max))
                else:
                    slc = ((0, H_min, W_min),
                            (c_in-1,H_max,W_max))

            elif isinstance(layer,Flatten):
                shape = layer_inp[1:]
                slc = (unflatten_unit(shape,slc[0]),unflatten_unit(shape,slc[1]))
            elif isinstance(layer,nn.Linear):
                shape = layer_inp[1:]
                slc = (tuple(0 for x in shape),tuple(x-1 for x in shape))
            elif layer_inp == layer_out:
                pass
            else:
                print ("Cannot compute dependency cone for layer of type {}.".format(type(layer)))
                exit(1)
            n1 -= 1
        return tuple(slice(x,y+1) for x,y in zip(slc[0],slc[1]))




    def get_cone(self,n,n1,slc) -> Tuple:
        #print("----------start-----------")
        model = self.model
        model.eval()
        slc_lists = [[] for i in range(n1-n+1)]
        slc_lists[-1] = [slc]
        while n1 > n:
            layer = self.get_layer(n1)
            layer_inp = self.layer_shape(n1-1)
            layer_out = self.layer_shape(n1)

            slcs = slc_lists[n1-n]
            for slc in slcs:
                if isinstance(layer,nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                    c_in = layer_inp[1]
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


                    H_min = max(0, slc[0][1]*stride[0]-padding[0])
                    W_min = max(0,slc[0][2]*stride[1]-padding[1])
                    H_max = min(H-1, slc[1][1]*stride[0]+dilation[0]*(kernel[0]-1)-padding[0])
                    W_max = min(W-1, slc[1][2]*stride[1]+dilation[1]*(kernel[1]-1)-padding[1])

                    if isinstance(layer, nn.MaxPool2d):
                        new_slc = ((slc[0][0], H_min, W_min),
                                (slc[1][0],H_max,W_max))
                    else:
                        new_slc = ((0, H_min, W_min),
                                (c_in-1,H_max,W_max))

                elif isinstance(layer,Flatten):
                    shape = layer_inp[1:]
                    new_slc = (unflatten_unit(shape,slc[0]),unflatten_unit(shape,slc[1]))
                elif isinstance(layer,nn.Linear):
                    shape = layer_inp[1:]
                    new_slc = (tuple(0 for x in shape),tuple(x-1 for x in shape))
                elif layer_inp == layer_out:
                    pass
                else:
                    print ("Cannot compute dependency cone for layer of type {}.".format(type(layer)))
                    exit(1)
                #layer_id = id(layer)
                for i in range(self.n):
                    if E[i][n1]:
                        slc_lists[i-n].append(new_slc)


            n1 -= 1
        return tuple(slice(x,y+1) for x,y in zip(slc[0],slc[1]))





