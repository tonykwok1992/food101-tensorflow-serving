FROM  tensorflow/serving:1.13.1
ADD classifier /classifier
ENTRYPOINT ["tensorflow_model_server", "--model_base_path=/classifier", "--rest_api_port=9000", "--model_name=food101"]


