## Deploy a mxnet model by lambda

- pros: simple, least management, low cost
- cons: cold start as the ecr image size about 500MB
- mxnet mnist dataset
- build and train a simple mxnet nn model locally
- upload the model (params) to s3
- deploy the model into lambda and apigaw

## Lambda processing code

it create a model and loads pre-trained params from a s3 bucket, then it perform prediction on an image passed by image url from the client request via apigw.

```py
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
    # prediction
    return {
            "statusCode": "200",
            "message": "hello mxnet"
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
    proxy: false,
    allowTestInvoke: false,
    credentialsRole: role,
    passthroughBehavior:
      cdk.aws_apigateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
    requestTemplates: {
      "application/json": fs.readFileSync(
        path.resolve(__dirname, "./../../mxnet-lambda/lambda-request-template"),
        {
          encoding: "utf-8",
        }
      ),
    },
    requestParameters: {},
    integrationResponses: [
      {
        statusCode: "200",
      },
    ],
  }),
  {
    methodResponses: [
      {
        statusCode: "200",
      },
    ],
  }
);
```

the request template (passthrough), can go to aws console and configure a passthrough request template.
