from configs.base.coral import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='FCN8s',
        params=dict(
            pretrained=True,
            classes=2,
            loss=dict(
                ignore_index=-1,
                ce=dict(),
            ),
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)