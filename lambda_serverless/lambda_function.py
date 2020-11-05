import os
import io
import boto3
import json

import csv

sagemaker = boto3.client('sagemaker')
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def response_object_creation():
    response_object = {}
    response_object['statusCode'] = 200
    response_object['headers'] = {
        'Content-Type': 'application/json'
    }
    return response_object


def lambda_handler(event, context):
    
    runtime= boto3.client('runtime.sagemaker')
    data = json.loads(json.dumps(event))
    payload = data['body']
    
    result_object = {}
    
    if(not payload):
        response_object = response_object_creation()
        response_object['body'] = 'Insufficient data'
        return response_object
        
      
    if(not event['key']):
        response = runtime.invoke_endpoint(EndpointName= ENDPOINT_NAME ,Body= payload, ContentType='text/csv', Accept = 'application/json')
        result = json.loads(response['Body'].read().decode())
        response_object = response_object_creation()
        response_object['body'] = result
        return response_object
    
        
    
    s3 = boto3.client('s3')
    csv1_data = s3.get_object(Bucket='sagemaker-us-east-2-326300720983', Key= event['key'])
    
    lines = csv1_data['Body'].read().decode("utf-8")
    
    payload_full = ''
    for line in lines.splitlines():
        payload_full+=line
        payload_full+=str('\n')
    
    response_csv = runtime.invoke_endpoint(EndpointName= ENDPOINT_NAME,Body= payload_full, ContentType='text/csv', Accept = 'application/json')
    result_csv = json.loads(response_csv['Body'].read().decode())
    
   
    response_object = response_object_creation()
    response_object['body'] = result_csv
    return response_object
    
    
    
        
                                   
                                       
        
   
