"""
A simple example showing how to construct an ObservationEncoder for processing multiple input modalities.
This is purely for instructional purposes, in case others would like to make use of or extend the
functionality.
"""
import textwrap
import torch
import numpy as np
import torch.nn as nn
from robomimic.models.base_nets import ConvBase
from robomimic.models.base_nets import *

class LCPMConv(ConvBase):
    """
    Base class for ConvNets pretrained with LCPM
    """
    def __init__(
        self,
        input_channel=3,
        r3m_model_class='resnet18',
        freeze=True,
    ):
        """
        Using LCPM pretrained observation encoder network
        Args:
            input_channel (int): number of input channels for input images to the network.
                If not equal to 3, modifies first conv layer in ResNet to handle the number
                of input channels.
            LCPM_model_class (str): select one of the r3m pretrained model "resnet18", "resnet34" or "resnet50"
            freeze (bool): if True, use a frozen R3M pretrained model.
        """
        super(LCPMConv, self).__init__()

        try:
            from LCPM import load_lcpm
        except ImportError:
            print("WARNING: could not load r3m library! Please follow https://github.com/facebookresearch/r3m to install R3M")

        net = load_lcpm()

        assert input_channel == 3 # R3M only support input image with channel size 3
        assert r3m_model_class in ["resnet18", "resnet34", "resnet50"] # make sure the selected r3m model do exist

        # cut the last fc layer
        self._input_channel = input_channel
        self._r3m_model_class = r3m_model_class
        self._freeze = freeze
        self._input_coord_conv = False
        self._pretrained = True

        # preprocess = nn.Sequential(
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # )
        # self.nets = Sequential(*([preprocess] + list(net.module.convnet.children())), has_output_shape = False)
        self.nets = net
        # if freeze:
        #     self.nets.freeze()

        self.weight_sum = np.sum([param.cpu().data.numpy().sum() for param in self.nets.parameters()])
        if freeze:
            for param in self.nets.parameters():
                param.requires_grad = False

        self.nets.train()

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module.
        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend
                on the size of the input, or if they assume fixed size input.
        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        assert(len(input_shape) == 3)

        out_dim = 512
        # if self._r3m_model_class == 'resnet50':
        #     out_dim = 2048
        # else:
        #     out_dim = 512

        return [out_dim, 1, 1]

    def forward(self, inputs):
        obs = inputs
        bs,_, _, _ = obs.shape
        lang = ["Pick up a can"]*bs
        # lang = inputs["lang"]
        x = self.nets(obs, lang)
        if list(self.output_shape(list(inputs.shape)[1:])) != list(x.shape)[1:]:
            raise ValueError('Size mismatch: expect size %s, but got size %s' % (
                str(self.output_shape(list(inputs.shape)[1:])), str(list(x.shape)[1:]))
            )
        return x
    
    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        return header + '(input_channel={}, input_coord_conv={}, pretrained={}, freeze={})'.format(self._input_channel, self._input_coord_conv, self._pretrained, self._freeze)

