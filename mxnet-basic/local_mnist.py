import sys
from mxnet import nd
import mxnet as mx
from multiprocess import cpu_count
from mxnet import nd
from mxnet.gluon import nn
from mxnet import autograd


def test_dataloader():
    # cpu count
    print(f"cpu count {cpu_count()}")
    X = mx.random.uniform(shape=(10, 3))
    y = mx.random.uniform(shape=(10, 1))
    dataset = mx.gluon.data.dataset.ArrayDataset(X, y)
    data_loader = mx.gluon.data.DataLoader(
        dataset,
        batch_size=5,
        num_workers=cpu_count(),
    )
    print(sys.getsizeof(data_loader))
    # actual data
    for X_batch, y_batch in data_loader:
        print(
            f"X_bach has shape {X_batch.shape} and y_batch shape {y_batch.shape}"
        )
        print(y_batch)


def data_xform(data):
    return nd.moveaxis(data, 2, 0).astype("float32") / 255.0


def load_data():
    # load data
    train_data = mx.gluon.data.vision.MNIST(train=True).transform_first(
        data_xform
    )
    val_data = mx.gluon.data.vision.MNIST(train=False).transform_first(
        data_xform
    )

    # dataloader - threadpool worker to load (x,y) samples
    batch_size = 100
    train_loader = mx.gluon.data.DataLoader(
        train_data, shuffle=True, batch_size=batch_size
    )
    val_loader = mx.gluon.data.DataLoader(
        val_data, shuffle=False, batch_size=batch_size
    )
    # print actual data
    # for x, y in train_loader:
    #     print(y)
    return train_loader, val_loader


def build_model():
    net = nn.Sequential()
    with net.name_scope():
        net.add(
            nn.Flatten(),
            nn.Dense(128, activation="relu"),
            nn.Dense(64, activation="relu"),
            nn.Dense(10, activation=None),
        )
    return net


def train_model():
    # create the modle
    net = build_model()
    # init parameters
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
    net.initialize(mx.init.Xavier(), ctx=ctx)
    # trainer
    trainer = mx.gluon.Trainer(
        params=net.collect_params(),
        optimizer="sgd",
        optimizer_params={"learning_rate": 0.04},
    )
    # loss
    metric = mx.metric.Accuracy()
    loss_function = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    # training data
    train_loader, label_loader = load_data()
    #
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")
        for inputs, labels in train_loader:
            inputs = inputs.as_in_context(ctx)
            labels = labels.as_in_context(ctx)
            # record loss
            with autograd.record():
                outputs = net(inputs)
                loss = loss_function(outputs, labels)
            # backprop
            loss.backward()
            metric.update(labels, outputs)
            # update parameters of the network
            trainer.step(batch_size=inputs.shape[0])
        #
        name, acc = metric.get()
        print(f"after epoch {epoch+1} :  {name} = {acc}")
        metric.reset()


if __name__ == "__main__":
    # test_dataloader()
    load_data()
    train_model()
