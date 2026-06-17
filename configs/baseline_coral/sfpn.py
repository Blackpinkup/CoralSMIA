import torch.nn as nn


from configs.base.coral import test, data

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
        type='SemanticFPN',
        params=dict(
            encoder=dict(
                name='resnet50',
                weights='imagenet',
                in_channels=3
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
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
            classes=2,
            loss=dict(
                ce=dict()
            )
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=dict(
      tta=False,
    )
)
