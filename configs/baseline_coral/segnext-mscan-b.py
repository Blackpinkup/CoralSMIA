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
        type='SegNext',
        params=dict(
            encoder=dict(
                init_cfg=dict(type='Pretrained', checkpoint='/home/biqi/code/business/LoveNAS-master/pretrain/segnext/mscan_b.pth'),
                embed_dims=[64, 128, 320, 512],
                depths=[3, 3, 12, 3],
                mlp_ratios=[8, 8, 4, 4],
                drop_rate=0.0,
                drop_path_rate=0.1,
                attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='BN', requires_grad=True)
            ),
            decoder=dict(
                in_channels=[128, 320, 512],
                in_index=[1, 2, 3],
                channels=256,
                ham_channels=256,
                ham_kwargs=dict(MD_R=16),
                dropout_ratio=0.1,
                num_classes=2,
                norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                align_corners=False,
            ),
            classes=2,
            loss=dict(
                ignore_index=-1,
            )
        )),
        data=data,
        optimizer=optimizer,
        learning_rate=learning_rate,
        train=train,
        test=test
)
