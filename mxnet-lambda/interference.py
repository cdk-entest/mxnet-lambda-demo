import mxnet as mx
import numpy as np
import os
import boto3

# environment variables
bucket = os.environ["BUCKET_NAME"]
key = os.environ["MODEL_PATH"]

# download the saved model from s3
boto3.resource("s3").Bucket(bucket).download_file(key, "/tmp/model.tar.gz")
# extract the model into lambda tmp dir
os.system("tar -xvf /tmp/model/tar.gz -C /tmp/")


def lambda_handler(event, context):
    """
    mxnet perform perdiction inside lambda
    """
    # load model
    ctx = mx.cpu()
    epoch = 10
    sym, args, aux = mx.model.load_checkpoint(
        "/tmp/image-classification", epoch
    )
    # load images
    fname = mx.test_utils.download(event["url"], dirname="/tmp/")
    img = mx.image.imread(fname)
    # convert image into rbg format
    img = mx.image.imresize(img, 224, 224)
    img = img.transpose((2, 0, 1))
    img = img.expand_dims(axis=0)
    img = img.asytpe(dtype="float32")
    args["data"] = img
    # softmax
    softmax = mx.nd.random_normal(shape=(1,))
    args["softmax_label"] = softmax
    # execut prediction using the model
    exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req="null")
    exe.forward()
    prob = exe.outputs[0].asnumpy()
    # remove single dimensional
    prob = np.squeeze(prob)
    # classification labels
    labels = ["ak47", "american-flag"]
    #
    a = np.argsort(prob)[::-1]
    # print output
    output = []
    for i in a[0:5]:
        output.append({"label": labels[i], "probability": str(prob[i])})
        print("probability {prob[i]} and class {labels[i]}")
    # return
    return {"records": output}
