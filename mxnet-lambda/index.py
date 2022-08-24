"""
haimtran  
simple mxnet model in lambda 
"""

import os
import json
import boto3
import mxnet as mx
from mxnet.gluon import nn
import numpy as np


# s3 client
s3 = boto3.resource("s3")

# download model params
s3.meta.client.download_file(
    Bucket=os.environ["BUCKET_NAME"],
    Key=os.environ["MODEL_PATH"],
    Filename="/tmp/model_params",
)

# create a nn model
def create_model():
    """
    create a nn model
    """

    net = mx.gluon.nn.Sequential()
    net.add(
        nn.Flatten(),
        nn.Dense(128, activation="relu"),
        nn.Dense(64, activation="relu"),
        nn.Dense(10, activation=None),
    )
    return net


# read image from file
def read_image(image_url):
    """
    read image from s3 and transform to mxnet format
    """
    #
    file_name = "/tmp/image.png"
    # get image from s3
    # TODO: read into buffer - not saving file
    s3.meta.client.download_file(
        Bucket=os.environ["BUCKET_NAME"], Key=image_url, Filename=file_name
    )
    # mxnet read image
    image = mx.image.imread(filename=file_name)
    # transform to mxnet format
    image = mx.image.imresize(image, 28, 28)
    image = image.transpose((2, 0, 1))
    image = image.astype(dtype="float32")
    # transform image to  mx format
    return image


def handler(event, context):
    """
    lambda handler
    """

    print(event)
    # create a model
    net = create_model()
    # load params into model
    net.load_parameters(filename="/tmp/model_params")
    # parse request
    image_url = event["queryStringParameters"]["image_url"]
    # read the image from s3
    image = read_image(image_url)
    # predict
    pred = net(image)[0]
    pred = pred.asnumpy()
    pred_dict = dict(zip(np.arange(10), pred))
    pred_dict = {
        k: v for k, v in sorted(pred_dict.items(), key=lambda item: item[1])
    }
    # prediction
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,GET",
        },
        "body": json.dumps({"pred": f"{pred_dict}"}),
    }


# local test
if __name__ == "__main__":
    resp = handler(
        event={"queryStringParameters": {"image_url": "images/image-1.jpg"}},
        context=None,
    )
    print(resp)
