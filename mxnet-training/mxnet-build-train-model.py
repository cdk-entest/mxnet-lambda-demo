"""
15 aug 2022 haimtran 
mxnet mnist train and save model locally
"""
from mxnet import nd
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from mxnet import autograd
import matplotlib.pyplot as plt
import numpy as np


def data_xform(data):
    """
    transform data
    """
    return nd.moveaxis(data, 2, 0).astype("float32") / 255.0


def load_data():
    """
    create dataloader
    """
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
    return train_loader, val_loader


def create_model():
    """
    create a model nn
    """
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
    """
    train the model
    """
    # create the modle
    net = create_model()
    # init parameters
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else [mx.cpu(0), mx.cpu(1)]
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
    # training
    train_loader, _ = load_data()
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
    # save model
    net.save_parameters("net.params")


def predict():
    """
    load model from params and predict
    """
    #
    metric = mx.metric.Accuracy()
    # context
    ctx = mx.gpu(0) if mx.context.num_gpus() > 0 else mx.cpu(0)
    # load val data
    _, val_data = load_data()
    # create the model again
    net = create_model()
    # load model with pre-trained params
    net.load_parameters("net.params")
    # predict
    outputs = []
    for inputs, labels in val_data:
        # gpu option
        # inputs = inputs.as_in_context(ctx)
        # labels = labels.as_in_context(ctx)
        # predict
        pred = net(inputs)
        # print
        # print(f"input: {inputs} and output: {pred}")
        metric.update(labels, pred)
        outputs.append(net(inputs))
        # break
    print(f"validation {metric.get()} ")


def read_mnist_images():
    """
    read image
    """
    _, val_data = load_data()
    k = 0
    for images, _ in val_data:
        image = images[0][0].asnumpy()
        plt.imshow(image, cmap="gray")
        plt.imsave(f"./../data/image-{k}.png", image, cmap="gray")
        k = k + 1
        if k > 10:
            break


def load_image_from_file():
    """
    use open cv to read image
    """
    image = mx.image.imread("./../data/image-2.png", 0)
    image = mx.image.imresize(image, 28, 28)
    image = image.transpose((2, 0, 1))
    image = image.astype(dtype="float32")
    return image


def test_local_image():
    """
    read image and test locally
    """
    # create model
    net = create_model()
    # load model
    net.load_parameters("net.params")
    # read an image from a file
    image_val = load_image_from_file()
    # predict
    pred = net(image_val)[0]
    pred = pred.asnumpy()
    temp = dict(zip(np.arange(10), pred))
    print({k: v for k, v in sorted(temp.items(), key=lambda item: item[1])})


if __name__ == "__main__":
    # test_local_image()
    read_mnist_images()
