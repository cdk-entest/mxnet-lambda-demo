from mxnet.gluon import nn
from mxnet import nd
from mxnet import autograd

# first layer
layer = nn.Dense(2)
print(layer)
layer.initialize()

# x input
x = nd.random.uniform(-1, 1, (3, 4))
print(layer(x))

# check layer weight
# weights randomly uniform within [-0.7, 0.7]
print(layer.weight.data())

# chain layer
net = nn.Sequential()
net.add(
    # similar to dense, not necessary to specify the input channels
    nn.Conv2D(channels=6, kernel_size=5, activation="relu"),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Conv2D(channels=16, kernel_size=3, activation="relu"),
    nn.MaxPool2D(pool_size=2, strides=2),
    nn.Dense(120, activation="relu"),
    nn.Dense(84, activation="relu"),
    nn.Dense(10),
)
# net summary
print(net.summary)

# initialize
net.initialize()
x = nd.random.uniform(shape=(4, 1, 28, 28))
y = net(x)
print(y.shape)
print(y)

# autograd
x = nd.array([[1, 2], [3, 4]])
x.attach_grad()
with autograd.record():
    y = 2 * x * x
# derivative
print(y)
print(y.backward())
print(x.grad)
