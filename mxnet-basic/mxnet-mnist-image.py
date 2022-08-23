# haimtran 20 aug 2022
# mxnet dataset download and store data

import mxnet as mx
import gzip
import numpy as np
import matplotlib.pyplot as plt

# download the dataset to ~/.mxnet/datasets/
train_dataset = mx.gluon.data.vision.datasets.FashionMNIST(train=True)

# read from gzip dataset
num_images = 10
image_size = 28
f = gzip.open("./../data/train-images-idx3-ubyte.gz")
f.read(16)
buf = f.read(image_size * image_size * num_images)
data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
# print data shape
print(np.shape(data))
data = data.reshape(num_images, image_size, image_size, 1)

# plot the image
for k in range(10):
    image = np.asarray(data[k]).squeeze()
    # print(image)
    plt.imshow(image)
    plt.savefig(f"image-{k}.png")
