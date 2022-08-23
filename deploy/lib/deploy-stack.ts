import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { Duration } from 'aws-cdk-lib';
import * as path from 'path';
import { Effect } from 'aws-cdk-lib/aws-iam';
import config from "./config.json";


export class DeployStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // lambda with dependencies 
    const func = new cdk.aws_lambda.Function(
      this,
      "mxnetLambda",
      {
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
            actions: ['s3:*']
          })
        ]
      }
    )

    // role for apigw to invoke lambda 
    const role = new cdk.aws_iam.Role(
      this,
      "InvokeLambdaRokeForApiGw",
      {
        roleName: "InvokeLambdaRokeForApiGw",
        assumedBy: new cdk.aws_iam.ServicePrincipal("apigateway.amazonaws.com")
      }
    )

    role.addToPolicy(
      new cdk.aws_iam.PolicyStatement({
        effect: Effect.ALLOW,
        resources: [func.functionArn],
        actions: ['lambda:InvokeFunction']
      })
    )

    const api = new cdk.aws_apigateway.RestApi(
      this,
      "ApiGw",
      {
        restApiName: "ApiGwMxnet",
      }
    )

    const resource = api.root.addResource(
      "predict"
    )

    resource.addMethod(
      "GET",
      new cdk.aws_apigateway.LambdaIntegration(
        func,
        {
          proxy: true,
          allowTestInvoke: false,
          credentialsRole: role,
          passthroughBehavior: cdk.aws_apigateway.PassthroughBehavior.WHEN_NO_TEMPLATES,
          requestTemplates: {},
          requestParameters: {},
          integrationResponses: [
            {
              statusCode: "200"
            }
          ]
        }
      ),
      {
        methodResponses: [
          {
            statusCode: "200"
          }
        ]
      }
    )
  }
}
