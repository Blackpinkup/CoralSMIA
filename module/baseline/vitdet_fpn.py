import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ever.interface import ERModule
from ever import registry
from module.baseline.base import AssymetricDecoder, FPN, default_conv_block
import math
from module.loss import SegmentationLoss
from segmentation_models_pytorch.encoders import get_encoder
from module.baseline.swin.swin_transformer import SwinTransformer
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
from detectron2.layers import CNNBlockBase, Conv2d, get_norm

class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        input_shapes=768,
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        norm="LN",
        square_pad=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        strides = [int(14 / scale) for scale in scale_factors]
        # _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        features = x
        results = []

        for stage in self.stages:
            results.append(stage(features))

        assert len(self._out_features) == len(results)
        # return {f: res for f, res in zip(self._out_features, results)}
        return results

@registry.MODEL.register('ViTDetFPN')
class ViTDetFPN(ERModule):
    def __init__(self, config):
        super(ViTDetFPN, self).__init__(config)
        if 'swin' in self.config.encoder:
            self.en = SwinTransformer(**self.config.encoder.swin)
        if 'type' in self.config.encoder.keys():
            self.en = registry.MODEL[self.config.encoder.type](self.config.encoder.params)
        else:
            self.en = get_encoder(**self.config.encoder)
        # self.fpn = FPN(**self.config.fpn)
        self.decoder = AssymetricDecoder(**self.config.decoder)
        self.cls_pred_conv = nn.Conv2d(self.config.decoder.out_channels, self.config.classes, 1)
        #self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=4)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cls_loss = SegmentationLoss(self.config.loss)

        self.vitdet_fpn = SimpleFeaturePyramid(input_shapes=self.config.fpn['in_channels_list'][-1])


    def forward(self, x, y=None):
        if 'swin' in self.config.encoder:
            feat_list = self.en(x)
        else:
            feat_list = self.en(x)[1:]
        if 'type' in self.config.encoder.keys():
            if self.config.encoder['type'] == 'DinoV2Encoder':
                feat_list = self.en(x)
        # fpn_feat_list = self.fpn(feat_list)
        fpn_feat_list = self.vitdet_fpn(feat_list[-1])
        final_feat = self.decoder(fpn_feat_list)
        cls_pred = self.cls_pred_conv(final_feat)
        #cls_pred = self.upsample4x_op(cls_pred)
        cls_pred = F.interpolate(cls_pred, size=(x.shape[-2:]), mode='bilinear', align_corners=False)
        if self.training:
            cls_true = y['cls']
            #loss_dict = dict()
            loss = self.cls_loss(cls_pred, cls_true)
            #mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            #loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss
        else:
            if 'mbloss' in self.config.loss:
                cls_prob = torch.sigmoid(cls_pred)
            else:
                cls_prob = torch.softmax(cls_pred, dim=1)
            return cls_prob



    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(
                name='resnet50',
                weights='imagenet',
                in_channels=3
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                conv_block=default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            classes=7,
            loss=dict(
                ignore_index=255,
            )
        ))


