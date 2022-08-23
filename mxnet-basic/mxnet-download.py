import boto3
import mxnet as mx

# download the model from s3
boto3.resource("s3").Bucket("sagemaker-us-east-1-305047569515").download_file(
    Key="mxnet-mnist-example/code/mxnet-training-2022-08-18-07-35-42-297/source/sourcedir.tar.gz",
    Filename="sourcedir.tar.gz",
)

# download with mxnet
mx.test_utils.download(url="", dirname="./")
