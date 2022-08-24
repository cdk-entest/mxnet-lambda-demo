---
title: Host a ML Model in AWS Lambda
description: build and train a mxnet nn locally and host in aws lambda
author: haimtran
publishedDate: 08/20/2022
date: 2022-08-20
---

## Host a mxnet model aws lambda

[GitHub](whttps://github.com/entest-hai/mxnet-lambda-demo) this is part 1 of my series on SageMaker. I start by hosting a mxnet nn model in AWS Lambda and serving it via API Gateway.

- pros: simple, least management, low cost
- cons: cold start as the ecr image size about 500MB
- mxnet mnist dataset
- build and train a simple mxnet nn model locally
- upload the model (params) to a s3 bucket
- deploy the model into lambda and apigaw

<LinkedImage
  href="https://youtu.be/2xKhupRU0_4"
  height={400}
  alt="Host a ML Model in AWS Lambda"
  src="https://user-images.githubusercontent.com/20411077/186332154-6557b7dd-2ee7-4c20-9bbc-f889669f190b.png"
/>

## Lambda processing code

it create a model and loads pre-trained params from a s3 bucket, then it perform prediction on an image passed by image url from the client request via apigw.

```py
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
        Bucket=os.environ["BUCKET_NAME"],
        Key=image_url,
        Filename=file_name
    )
    # mxnet read image
    image = mx.image.imread(filename=file_name)
    # transform to mxnet format
    image = mx.image.imresize(image, 28, 28)
    image = image.transpose((2, 0, 1))
    image = image.astype(dtype="float32")
    # transform image to  mx format
    return image
```

then the lambda handler

```py
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
```

## Infrastructure deployment

lambda ecr

```tsx
const func = new cdk.aws_lambda.Function(this, "mxnetLambda", {
  functionName: "mxnetLambda",
  code: cdk.aws_lambda.Code.fromAssetImage(
    path.join(__dirname, "./../../mxnet-lambda")
  ),
  runtime: cdk.aws_lambda.Runtime.FROM_IMAGE,
  handler: cdk.aws_lambda.Handler.FROM_IMAGE,
  memorySize: 2048,
  timeout: Duration.seconds(300),
  environment: {
    BUCKET_NAME: config.BUCKET_NAME,
    MODEL_PATH: config.MODEL_PATH,
  },
  initialPolicy: [
    new cdk.aws_iam.PolicyStatement({
      effect: Effect.ALLOW,
      resources: ["*"],
      actions: ["s3:*"],
    }),
  ],
});
```

Dockerfile

```
FROM public.ecr.aws/lambda/python:3.7

# create code dir inside container
RUN mkdir ${LAMBDA_TASK_ROOT}/source

# copy code to container
COPY "requirements.txt" ${LAMBDA_TASK_ROOT}/source

# copy handler function to container
COPY ./index.py ${LAMBDA_TASK_ROOT}

# install dependencies for running time environment
RUN pip3 install -r ./source/requirements.txt --use-feature=2020-resolver --target "${LAMBDA_TASK_ROOT}"

# set the CMD to your handler
CMD [ "index.handler" ]

```

requirements, should be careful with dependencies for mxnet on the lambda.

```
mxnet==1.6.0
numpy==1.19.5
certifi==2020.6.20
idna==2.10
graphviz==0.8.4
chardet==3.0.4
```

lambda apigw integration, alternatively can use lambda proxy mode. The important thing is client will request with an image url, and apigw should pass this image url to the lambda.

```tsx
resource.addMethod(
  "GET",
  new cdk.aws_apigateway.LambdaIntegration(func, {
    proxy: true,
    allowTestInvoke: false,
    credentialsRole: role,
  })
);
```

## Tsx load json

modify tsconfig.js to load config.json

```json
 "resolveJsonModule": true,
    "esModuleInterop": true,
```
