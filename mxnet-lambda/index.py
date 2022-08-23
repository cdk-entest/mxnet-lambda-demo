import os
import json
import boto3
import mxnet as mx
from mxnet.gluon import nn
import numpy as np


# download model params from s3 into a file
boto3.resource('s3').Bucket(os.environ['BUCKET_NAME']).download_file(Key=os.environ['MODEL_PATH'], Filename="/tmp/model_params")

def create_model():
    net = mx.gluon.nn.Sequential()
    net.add(
            nn.Flatten(),
            nn.Dense(128, activation='relu'),
            nn.Dense(64, activation='relu'),
            nn.Dense(10, activation=None)
            )
    return net 

def handler(event, context):
    print(event)
    # create a model 
    net = create_model()
    # load params into model 
    net.load_parameters(
            filename="/tmp/model_params"
            )
    # parse request  
    # image_url = event['params']['querystring']['image_url']
    # read image 
    # image = mx.image.imread(filename=image_url)
    # transform image 
    # image = image 
    # model predict  
    # pred = net(image)
    # softmax output 

    # prediction
    return {
        'statusCode': 200,
        'headers': {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,GET"
        },
        'body': json.dumps({
            'message': f'{event}'
        })
    }
