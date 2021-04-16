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
from .model_mgr import unflatten_unit
from nnitp.models.models import Flatten

# This class is the interface to torch models.



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

    # Constructor from a Keras model.

    def __init__(self,model, inp_shape):
        self.model = model
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

    def init_shape(self):
        for i,l in enumerate(self._layers):
            #self.fhooks.append(l.register_forward_hook(self.forward_hook(i)))
            self.shape_hooks.append(l.register_forward_hook(self.shape_hook(i)))
        x = torch.randn(self.inp_shape)
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
            self.mid_out = out.detach()
        return hook

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


    def compute_activation(self,lidx,test):
        self.model.eval()
        if not torch.is_tensor(test):
            test = torch.tensor(test)
        if lidx < 0:
            return test
        l = self.get_layer(lidx)
        self.fhook = l.register_forward_hook(self.forward_hook(lidx))
        with torch.no_grad():
          self.model(test)
        self.fhook.remove()
        return self.mid_out
        #inp = self.model.input
        #functor = K.function([inp, K.learning_phase()], [self.model.layers[lidx].output] )
        #return functor([test])[0]

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
        model = self.model
        while n1 > n:
            layer = self.get_layer(n1)
            layer_inp = self.layer_shape(n1-1)
            layer_out = self.layer_shape(n1)
            if isinstance(layer,nn.Conv2d) or isinstance(layer, nn.MaxPool2d):
                #weights = layer.get_weights()[0]
                #out, inp, row_size,col_size = weights.shape[i] for in len(weights.shape)
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


                H_min = slc[0][0]*stride[0]-padding[0]
                W_min = slc[0][1]*stride[1]-padding[1]
                H_max = slc[1][0]*stride[0]+dilation[0]*(kernel[0]-1)-padding[0]
                W_max = slc[1][1]*stride[1]+dilation[1]*(kernel[1]-1)-padding[1]
                #shp = layer.input_shape
                #shp = self.layer_shape(n1-1)
                #if layer.padding == 'same':
                #    rp = -(row_size // 2)
                #    cp = -(col_size // 2)
                #    slc = ((max(0,slc[0][0]+rp),max(0,slc[0][1]+cp),0),
                #           (min(shp[1]-1,slc[1][0]+row_size-1+cp),
                #            min(shp[2]-1,slc[1][1]+col_size-1+cp),planes-1))
                #else:
                #    slc = ((slc[0][0],slc[0][1],0),
                #           (slc[1][0]+row_size-1,slc[1][1]+col_size-1,planes-1))
            #elif isinstance(layer,MaxPooling2D):
            #    wrows,wcols = layer.pool_size
                #def foo(i):
                #    return (slc[i][0] * wrows,slc[i][1] * wcols,slc[i][2])
            #    slc = ((slc[0][0] * wrows,slc[0][1] * wcols,slc[0][2]),
            #           ((slc[1][0]+1)*wrows-1,(slc[1][1]+1)*wcols-1,slc[1][2]))
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


