# Copyright (c) OpenMMLab. All rights reserved.
from mmocr.models.builder import HEADS
from . import PANHead


@HEADS.register_module()
class PSEHead(PANHead):
    """The class for PSENet head.

    Args:
        in_channels (list[int]): A list of 4 numbers of input channels.
        out_channels (int): Number of output channels.
        text_repr_type (str): Use polygon or quad to represent. Available
            options are "poly" or "quad".
        downsample_ratio (float): Downsample ratio.
        loss (dict): Configuration dictionary for loss type. Supported loss
            types are "PANLoss" and "PSELoss".
        train_cfg, test_cfg (dict): Depreciated.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            text_repr_type='poly',  # 'poly' or 'quad'
            downsample_ratio=0.25,
            loss=dict(type='PSELoss'),
            train_cfg=None,
            test_cfg=None,
            init_cfg=None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            text_repr_type=text_repr_type,
            downsample_ratio=downsample_ratio,
            loss=loss,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
