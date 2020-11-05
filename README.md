# DocumentClassificationBkfs

Deploy Amazon Sagemaker model

1. Access Amazon sagemaker on AWS. Create notebook instances and copy the files given under classifier

2. Open, AWS-sagemaker.ipynb, follow the jupyter code and execute it. After executing, you will receive a model at an endpoint.

3. As provided in the notebook, execute the request using boto client to get a sample response.


Deploy Lambda

1. Access Amazon Lambda, create a new lambda function, and copy the content provided under lambda_function.py
2. Save and press deploy button on Inline code editor and test the code.
3. Access API gateway, create a new rest api using the above lambda function. Configure the resource, add method and deploy the api (Stage)
4. Deploy the api and test the webservice using postman. Sample request is given in requests.txt file.

Url: https://fku4xsr83b.execute-api.us-east-2.amazonaws.com/test/predict


Deploy App to S3

This is a work in progress. Faced a few issues with the ajax post request from frontend. I will update soon.








