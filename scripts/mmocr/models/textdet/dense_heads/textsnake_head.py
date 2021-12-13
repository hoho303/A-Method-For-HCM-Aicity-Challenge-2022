# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from mmocr.models.builder import HEADS, build_loss
from . import HeadMixin


@HEADS.register_module()
class TextSnakeHead(HeadMixin, BaseModule):
    """The class for TextSnake head.

    TextSnake: `A Flexible Representation for Detecting Text of Arbitrary
    Shapes <https://arxiv.org/abs/1807.01544>`_.

    Args:
        in_channels (int): Number of input channels.
        decoding_type (str): Decoding type. It usually should not be changed.
        text_repr_type (str): Use polygon or quad to represent. Available
            options are "poly" or "quad".
        loss (dict): Configuration dictionary for loss type.
        train_cfg, test_cfg: Depreciated.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels,
                 decoding_type='textsnake',
                 text_repr_type='poly',
                 loss=dict(type='TextSnakeLoss'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     override=dict(name='out_conv'),
                     mean=0,
                     std=0.01)):
        super().__init__(init_cfg=init_cfg)

        assert isinstance(in_channels, int)
        self.in_channels = in_channels
        self.out_channels = 5
        self.downsample_ratio = 1.0
        self.decoding_type = decoding_type
        self.text_repr_type = text_repr_type
        self.loss_module = build_loss(loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.out_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        """
        Args:
            inputs (Tensor): Shape :math:`(N, C_{in}, H, W)`, where
                :math:`C_{in}` is ``in_channels``. :math:`H` and :math:`W`
                should be the same as the input of backbone.

        Returns:
            Tensor: A tensor of shape :math:`(N, 5, H, W)`.
        """
        outputs = self.out_conv(inputs)
        return outputs
