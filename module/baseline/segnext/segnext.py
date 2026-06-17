import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ever.interface import ERModule
from ever import registry
import math
from transformers import AutoImageProcessor, ViTModel
device = "cuda" if torch.cuda.is_available() else "cpu"
from module.baseline.segnext.mscan import MSCAN
from module.baseline.segnext.ham_head import LightHamHead



@registry.MODEL.register('SegNext')
class SegNext(ERModule):
    def __init__(self, config):
        super(SegNext, self).__init__(config)
        self.encoder = MSCAN(**self.config.encoder)
        self.decoder = LightHamHead(**self.config.decoder)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=8)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)


    def forward(self, x, y=None):
        outs = self.encoder(x)
        cls_out = self.decoder(outs)
        cls_pred = self.upsample4x_op(cls_out)
        if self.training:
            #cls_true = y['mask'].long()
            cls_true = y['cls'].long()
            loss_dict = dict()
            loss_dict['seg_loss'] = self.ce_loss(cls_pred, cls_true)
            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict
        else:
            return cls_pred.softmax(dim=1)



    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(
                init_cfg=dict(type='Pretrained', checkpoint='/home/biqi/code/business/LoveNAS-master/pretrain/segnext/mscan_b.pth'),
                embed_dims=[32, 64, 160, 256],
                mlp_ratios=[8, 8, 4, 4],
                drop_rate=0.0,
                drop_path_rate=0.1,
                depths=[3, 3, 5, 2],
                attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='BN', requires_grad=True)
            ),

            decoder=dict(
                in_channels=[64, 160, 256],
                in_index=[1, 2, 3],
                channels=256,
                ham_channels=256,
                dropout_ratio=0.1,
                num_classes=2,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                align_corners=False,
            ),
            classes=2,
            loss=dict(
                ignore_index=-1,
            )
        ))

if __name__ == '__main__':
    m = SegNext(dict()).eval()
    image = torch.ones(1, 3, 512, 512)
    l = m(image)
    print(l.shape)