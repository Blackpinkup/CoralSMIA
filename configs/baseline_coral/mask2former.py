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
        type='Mask2Former',
        params=dict(
            pretrain_dir='/home/biqi/code/business/LoveNAS-master/pretrain/mask2former-swin-base-IN21k-cityscapes-semantic/'
        )),
        data=data,
        optimizer=optimizer,
        learning_rate=learning_rate,
        train=train,
        test=test
    )