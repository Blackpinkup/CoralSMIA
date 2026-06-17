from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop, RandomResizedCrop, Affine, PadIfNeeded, Resize
import ever as er
from ever.api.preprocess.albu import RandomDiscreteScale
import cv2
from configs.base.coral import test

data = dict(
    train=dict(
        type='CoralLoader',
        params=dict(
            image_dir=[
                '/home/biqi/code/business/LoveNAS-master/dataset/CoralMask/train/images_re/',
            ],
            mask_dir=[
                '/home/biqi/code/business/LoveNAS-master/dataset/CoralMask/train/masks_re/',
            ],
            transforms=Compose([
                #RandomDiscreteScale([0.5, 0.75, 1.0, 1.25, 1.5, 1.75]),
                #RandomCrop(512, 512),
                Affine(scale=(0.5, 1.75), p=1.0),
                PadIfNeeded(min_height=518, min_width=518, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
                RandomCrop(height=518, width=518, p=1.0),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ], is_check_shapes=False),
            CV=dict(k=10, i=-1),
            training=True,
            batch_size=16,
            num_workers=4,
        ),
    ),
    test=dict(
        type='CoralLoader',
        params=dict(
            image_dir=[
                '/home/biqi/code/business/LoveNAS-master/dataset/CoralMask/test/images_re/',
            ],
            mask_dir=[
                '/home/biqi/code/business/LoveNAS-master/dataset/CoralMask/test/masks_re/',
            ],
            transforms=Compose([
                Resize(518, 518),
                Normalize(mean=(123.675, 116.28, 103.53),
                          std=(58.395, 57.12, 57.375),
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()

            ], is_check_shapes=False),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=2,
            num_workers=2,
        ),
    ),
)

optimizer = dict(
    type='adamw',
    params=dict(
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.0001,
    ),
)

learning_rate = dict(
    type='cosine',
    params=dict(
        base_lr=1e-4,
        max_iters=15000,
        eta_min=1e-6,
    ))
    
train = dict(
    forward_times=1,
    num_iters=15000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=30,
    eval_interval_epoch=30,
)

config = dict(
    model=dict(
        type='ViTDetFPN',
        params=dict(
            encoder=dict(
                type='DinoV2Encoder',
                params=dict(
                    name='dinov2_vitb14_reg',
                    params=dict(
                        img_size=(518, 518),
                        pretrained=True,
                        out_indices=(2, 5, 8, 11),
                    )
                ),
            ),
            fpn=dict(
                in_channels_list=(768, 768, 768, 768),
                out_channels=256,
                scales=(4.0, 2.0, 1.0, 0.5),
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(18, 37, 74, 148),
                out_feat_output_stride=9,
                num_groups_gn=None
            ),
            classes=2,
            loss=dict(
                ignore_index=-1,
                ce=dict(),
            )
        )),
        data=data,
        optimizer=optimizer,
        learning_rate=learning_rate,
        train=train,
        test=test
    )
