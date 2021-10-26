#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import os
import wget
from zipfile import ZipFile
import torch
import torch.autograd as autograd

from .mannequin_challenge.models import pix2pix_model
from .mannequin_challenge.options.train_options import TrainOptions
from .depth_model import DepthModel, DepthModelMeta

from torch.nn import functional as F
import torch.nn as nn

def get_model_from_url(url: str, local_path: str, is_zip: bool = False, path_root: str = "checkpoints") -> str:
    local_path = os.path.join(path_root, local_path)
    if os.path.exists(local_path):
        print(f"Found cache {local_path}")
        return local_path

    # download
    local_path = local_path.rstrip(os.sep)
    download_path = local_path if not is_zip else f"{local_path}.zip"
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    if os.path.isfile(download_path):
        print(f"Found cache {download_path}")
    else:
        print(f"Dowloading {url} to {download_path} ...")
        wget.download(url, download_path)

    if is_zip:
        print(f"Unziping {download_path} to {local_path}")
        with ZipFile(download_path, 'r') as f:
            f.extractall(local_path)
        os.remove(download_path)

    return local_path

class SuppressedStdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exception_type, exception_value, traceback):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
class MannequinChallengeModel(DepthModel):
    # Requirements and default settings
    align = 16
    learning_rate = 0.0004
    lambda_view_baseline = 0.1

    def __init__(self):
        super().__init__()

        parser = TrainOptions()
        parser.initialize()
        params = parser.parser.parse_args(["--input", "single_view"])
        params.isTrain = False

        model_file = get_model_from_url(
            "https://storage.googleapis.com/mannequinchallenge-data/checkpoints/best_depth_Ours_Bilinear_inc_3_net_G.pth",
            "mc.pth", path_root=os.path.dirname(__file__)
        )

        class FixedMcModel(pix2pix_model.Pix2PixModel):
            # Override the load function, so we can load the snapshot stored
            # in our specific location.
            def load_network(self, network, network_label, epoch_label):
                return torch.load(model_file)

        with SuppressedStdout():
            self.model = FixedMcModel(params)

    def train(self):
        self.model.switch_to_train()

    def eval(self):
        self.model.switch_to_eval()

    def parameters(self):
        return self.model.netG.parameters()

    def estimate_depth(self, images):
        images = autograd.Variable(images.cuda(), requires_grad=False)

        # Reshape ...CHW -> XCHW
        shape = images.shape
        C, H, W = shape[-3:]
        images = images.reshape(-1, C, H, W)

        self.model.prediction_d, _ = self.model.netG.forward(images)

        # Reshape X1HW -> BNHW
        out_shape = shape[:-3] + self.model.prediction_d.shape[-2:]
        self.model.prediction_d = self.model.prediction_d.reshape(out_shape)

        self.model.prediction_d = torch.exp(self.model.prediction_d)
        self.model.prediction_d = self.model.prediction_d.squeeze(-3)

        return self.model.prediction_d

    def save(self, file_name):
        state_dict = self.model.netG.state_dict()
        torch.save(state_dict, file_name)



class MannequinChallengeModelMeta(DepthModelMeta):
    # Requirements and default settings
    align = 16
    learning_rate = 0.0004
    lambda_view_baseline = 0.1

    def __init__(self):
        super().__init__()

        num_input = 3

        model_file = get_model_from_url(
            "https://storage.googleapis.com/mannequinchallenge-data/checkpoints/best_depth_Ours_Bilinear_inc_3_net_G.pth",
            "mc.pth", path_root=os.path.dirname(__file__)
        )
        state_dict = torch.load(model_file)
        self.vars, self.vars_bn = self.dict_to_parameter(state_dict)

        self.model = HourglassModelMeta(num_input)

    def dict_to_parameter(self, dict):
        vars = nn.ParameterDict()
        vars_bn = nn.ParameterDict()

        for key in dict:
            key_underbar = key.replace('.', '_')
            if key.startswith('uncertainty_layer'):
                continue
            if key.endswith('running_mean') or key.endswith('running_var'):
                vars_bn.update({key_underbar: nn.Parameter(data=dict[key], requires_grad=False)})
            else:
                vars.update({key_underbar: nn.Parameter(data=dict[key], requires_grad=True)})

        return vars, vars_bn

    def train(self):
        self.model.train()
        self.bn_training = True
    def eval(self):
        self.model.eval()
        self.bn_training = False
    def parameters(self):
        return self.vars.values()

    def estimate_depth(self, images, vars):
        if vars == None:
            vars = self.vars

        images = autograd.Variable(images.cuda(), requires_grad=False)

        # Reshape ...CHW -> XCHW
        shape = images.shape
        C, H, W = shape[-3:]
        images = images.reshape(-1, C, H, W)

        prediction_d = self.model(images, vars, self.vars_bn, self.bn_training)

        # Reshape X1HW -> BNHW
        out_shape = shape[:-3] + prediction_d.shape[-2:]
        prediction_d = prediction_d.reshape(out_shape)

        prediction_d = torch.exp(prediction_d)
        prediction_d = prediction_d.squeeze(-3)

        return prediction_d


def Conv2dMeta(x, vars, prefix, stride, padding):
    #w, b = vars[0], vars[1]
    w = vars[prefix+"_weight"]
    b = vars[prefix+"_bias"]
    x = F.conv2d(x, w, b, stride=stride, padding=padding)

    return x
        
def BatchNorm2dMeta(x, vars, vars_bn, prefix, affine, bn_training):
    #w, b = vars[0], vars[1]
    if affine == True:
        w = vars[prefix+"_weight"]
        b = vars[prefix+"_bias"]
    else:
        w = None
        b = None
    #running_mean, running_var = bn_vars[0], bn_vars[1]
    running_mean = vars_bn[prefix+"_running_mean"]
    running_var = vars_bn[prefix+"_running_var"]

    x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)

    return x

def ReLUMeta(x):
    x = F.relu(x, inplace=True)

    return x

class inceptionMeta(nn.Module):
    def __init__(self, config):
        super(inceptionMeta, self).__init__()
    
        self.n_inception = len(config)

        self.padding_list = [0]
        for i in range(1, self.n_inception):
            filt = config[i][0]
            pad = int((filt-1)/2)

            self.padding_list.append(pad)

    def __repr__(self):
        return "inception"+str(self.config)

    def forward(self, x, vars, vars_bn, prefix, bn_training):        
        prefix = prefix+"_convs"
        x_cat = []
        # Base 1*1 conv layer.
        x_base = Conv2dMeta(x, vars, prefix+"_0_0", stride=1, padding=self.padding_list[0])
        x_base = BatchNorm2dMeta(x_base, vars, vars_bn, prefix+"_0_1", False, bn_training)
        x_base = ReLUMeta(x_base)
        x_cat.append(x_base)

        for i in range(1, self.n_inception):
            prefix_i = prefix+"_{}".format(i)
            x_i = Conv2dMeta(x, vars, prefix_i+"_0", stride=1, padding=0)
            x_i = BatchNorm2dMeta(x_i, vars, vars_bn, prefix_i+"_1", False, bn_training)
            x_i = ReLUMeta(x_i)
            
            x_i = Conv2dMeta(x_i, vars, prefix_i+"_3", stride=1, padding=self.padding_list[i])
            x_i = BatchNorm2dMeta(x_i, vars, vars_bn, prefix_i+"_4", False, bn_training)
            x_i = ReLUMeta(x_i)
            
            x_cat.append(x_i)

        return torch.cat(x_cat, dim=1) # Channel dimension


class Channels1Meta(nn.Module):
    def __init__(self):
        super(Channels1Meta, self).__init__()
        
        self.inception_1_1 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception_1_2 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        
        self.inception_2_1 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception_2_2 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception_2_3 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])

    
    def forward(self, x, vars, vars_bn, prefix, bn_training):
        prefix = prefix+"_list"


        x1 = self.inception_1_1(x, vars, vars_bn, prefix+"_0_0", bn_training)
        x1 = self.inception_1_2(x1, vars, vars_bn, prefix+"_0_1", bn_training)

        x2 = F.avg_pool2d(x, 2, 2, 0)
        x2 = self.inception_2_1(x2, vars, vars_bn, prefix+"_1_1", bn_training)
        x2 = self.inception_2_2(x2, vars, vars_bn, prefix+"_1_2", bn_training)
        x2 = self.inception_2_3(x2, vars, vars_bn, prefix+"_1_3", bn_training)
        x2 = F.upsample_bilinear(x2, scale_factor=2)

        return x1+x2

class Channels2Meta(nn.Module):
    def __init__(self):
        super(Channels2Meta, self).__init__()

        self.inception_1_1 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception_1_2 = inceptionMeta([[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]])

        self.inception_2_1 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception_2_2 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.channels1 = Channels1Meta()
        self.inception_2_3 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception_2_4 = inceptionMeta([[64], [3, 64, 64], [7, 64, 64], [11, 64, 64]])

    def forward(self, x, vars, vars_bn, prefix, bn_training):
        prefix = prefix+"_list"

        x1 = self.inception_1_1(x, vars, vars_bn, prefix+"_0_0", bn_training)
        x1 = self.inception_1_2(x1, vars, vars_bn, prefix+"_0_1", bn_training)

        x2 = F.avg_pool2d(x, 2, 2, 0)
        x2 = self.inception_2_1(x2, vars, vars_bn, prefix+"_1_1", bn_training)
        x2 = self.inception_2_2(x2, vars, vars_bn, prefix+"_1_2", bn_training)
        x2 = self.channels1(x2, vars, vars_bn, prefix+"_1_3", bn_training)
        x2 = self.inception_2_3(x2, vars, vars_bn, prefix+"_1_4", bn_training)
        x2 = self.inception_2_4(x2, vars, vars_bn, prefix+"_1_5", bn_training)
        x2 = F.upsample_bilinear(x2, scale_factor=2)

        return x1+x2


class Channels3Meta(nn.Module):
    def __init__(self):
        super(Channels3Meta, self).__init__()

        self.inception_2_1 = inceptionMeta([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])
        self.inception_2_2 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.channels2 = Channels2Meta()
        self.inception_2_3 = inceptionMeta([[64], [3, 32, 64], [5, 32, 64], [7, 32, 64]])
        self.inception_2_4 = inceptionMeta([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])

        self.inception_1_1 = inceptionMeta([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])
        self.inception_1_2 = inceptionMeta([[32], [3, 64, 32], [7, 64, 32], [11, 64, 32]])


    def forward(self, x, vars, vars_bn, prefix, bn_training):
        prefix = prefix+"_list"

        x2 = F.avg_pool2d(x, 2, 2, 0)
        x2 = self.inception_2_1(x2, vars, vars_bn, prefix+"_0_1", bn_training)
        x2 = self.inception_2_2(x2, vars, vars_bn, prefix+"_0_2", bn_training)
        x2 = self.channels2(x2, vars, vars_bn, prefix+"_0_3", bn_training)
        x2 = self.inception_2_3(x2, vars, vars_bn, prefix+"_0_4", bn_training)
        x2 = self.inception_2_4(x2, vars, vars_bn, prefix+"_0_5", bn_training)
        x2 = F.upsample_bilinear(x2, scale_factor=2)

        x1 = self.inception_1_1(x, vars, vars_bn, prefix+"_1_0", bn_training)
        x1 = self.inception_1_2(x1, vars, vars_bn, prefix+"_1_1", bn_training)


        return x1+x2


class Channels4Meta(nn.Module):
    def __init__(self):
        super(Channels4Meta, self).__init__()

        self.inception_2_1 = inceptionMeta([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])
        self.inception_2_2 = inceptionMeta([[32], [3, 32, 32], [5, 32, 32], [7, 32, 32]])
        self.channels3 = Channels3Meta()
        self.inception_2_3 = inceptionMeta([[32], [3, 64, 32], [5, 64, 32], [7, 64, 32]])
        self.inception_2_4 = inceptionMeta([[16], [3, 32, 16], [7, 32, 16], [11, 32, 16]])

        self.inception_1_1 = inceptionMeta([[16], [3, 64, 16], [7, 64, 16], [11, 64, 16]])

    def forward(self, x, vars, vars_bn, prefix, bn_training):
        prefix = prefix+"_list"

        x2 = F.avg_pool2d(x, 2, 2, 0)
        x2 = self.inception_2_1(x2, vars, vars_bn, prefix+"_0_1", bn_training)
        x2 = self.inception_2_2(x2, vars, vars_bn, prefix+"_0_2", bn_training)
        x2 = self.channels3(x2, vars, vars_bn, prefix+"_0_3", bn_training)
        x2 = self.inception_2_3(x2, vars, vars_bn, prefix+"_0_4", bn_training)
        x2 = self.inception_2_4(x2, vars, vars_bn, prefix+"_0_5", bn_training)
        x2 = F.upsample_bilinear(x2, scale_factor=2)

        x1 = self.inception_1_1(x, vars, vars_bn, prefix+"_1_0", bn_training)

        return x1+x2

class HourglassModelMeta(nn.Module):
    def __init__(self, num_input):
        super(HourglassModelMeta, self).__init__()

        self.channels4 = Channels4Meta()
    
    def forward(self, input, vars, vars_bn, bn_training):
        prefix = 'seq'

        x = Conv2dMeta(input, vars, prefix+'_0', stride=1, padding=3)
        x = BatchNorm2dMeta(x, vars, vars_bn, prefix+'_1', True, bn_training)
        x = ReLUMeta(x) # id=2

        x = self.channels4(x, vars, vars_bn, prefix+'_3', bn_training)

        x = Conv2dMeta(x, vars, "pred_layer", stride=1, padding=1)

        return x

