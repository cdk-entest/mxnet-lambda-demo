import os
from sagemaker.session import Session
from sagemaker.mxnet import MXNet
import boto3

# role for sagemaker
role = os.environ["ROLE_ARN"]

# s3 bucket for saving code and model artifacts
bucket = Session().default_bucket()

# bucket location where your custom code
custom_code_upload_location = f"s3://{bucket}/mxnet-mnist-example/code"

# bucket location results of model training
model_artifacts_location = f"s3://{bucket}/mxnet-mnist-example/code"

# role execution - no role from ec2-credentials
# role = get_execution_role()
# sagemaker estimator
mnist_estimator = MXNet(
    entry_point="mnist.py",
    py_version="py3",
    framework_version="1.4.1",
    hyperparameters={"learning-rate": 0.1},
    role=role,
    output_path=model_artifacts_location,
    code_location=custom_code_upload_location,
    instance_count=1,
    instance_type="ml.m4.xlarge",
)

# train and test data location
region = boto3.Session().region_name
train_data_location = f"s3://sagemaker-sample-data-{region}/mxnet/mnist/train"
test_data_location = f"s3://sagemaker-sample-data-{region}/mxnet/mnist/test"

# fit model
mnist_estimator.fit({"train": train_data_location, "test": test_data_location})
