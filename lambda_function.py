import boto3
import os

from utils import initialize_logger
from ModelBatchInference import ModelBatchInference
from dotenv import load_dotenv
load_dotenv()


def lambda_handler(event, context):
    data_inp_list = ["Write a binary sort code in python", 
                 "Create a simple Convolutional Neural Network using PyTorch library."]
    

    S3_BUCKET = os.environ["S3_BUCKET"]
    INP_DATA_DIR = os.environ["INP_DATA_DIR"]
    OUT_DATA_DIR = os.environ["OUT_DATA_DIR"]
    MODEL_NAME = os.environ["MODEL_NAME"]
    MODEL_URI = os.environ["MODEL_URI"]

    session = boto3.Session()
    logger = initialize_logger()

    env = {'HF_TASK': 'text-generation'}
    batch_inf = ModelBatchInference(model_name=MODEL_NAME,
                                model_uri=MODEL_URI,
                                instance_type="ml.g5.4xlarge",
                                instance_count=1,
                                aws_session=session,
                                s3_bucket=S3_BUCKET,
                                logger=logger,
                                data_inp_dir=INP_DATA_DIR,
                                data_out_dir=OUT_DATA_DIR,
                                env=env
                               )
    
    batch_inf.prepare_data(data_inp_list=data_inp_list)

    batch_inf.execute_batch_job(trans_version="4.37", pytorch_version="2.1",
                          py_version="py310")
