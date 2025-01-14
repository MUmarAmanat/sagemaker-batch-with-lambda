import json
import sagemaker

from io import BytesIO
from sagemaker.huggingface import HuggingFaceModel


class ModelBatchInference:
    def __init__(self, model_name, model_uri, instance_type, 
                 instance_count, aws_session, s3_bucket,
                 logger, data_inp_dir, data_out_dir, env):
        self.model_name = model_name
        self.model_uri = model_uri
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.session = aws_session
        self.logger = logger
        self.s3_bucket = s3_bucket
        self.data_inp_dir = data_inp_dir
        self.data_out_dir = data_out_dir
        self.sagemaker_session = sagemaker.Session(boto_session=self.session)
        self.role = sagemaker.get_execution_role(self.sagemaker_session)
        self.env = env


    def prepare_data(self, data_inp_list):
        s3_client = self.session.client("s3")
        with BytesIO() as memory_file:
            for inp in data_inp_list:
                data = {"inputs": inp}
                memory_file.write((json.dumps(data) + '\n').encode('utf-8'))

            memory_file.seek(0)

            s3_client.put_object(Bucket=self.s3_bucket,
                                 Key=f"{self.data_inp_dir}/input.json",
                                 Body=memory_file)

        self.logger.info(f"Input data directory {self.data_inp_dir}/input.json")
        self.logger.info(f"Input Data for Batch inference has been saved at {self.s3_bucket}/{self.data_inp_dir}")


    def execute_batch_job(self, trans_version, pytorch_version,
                          py_version):

        output_s3_path = f"s3://{self.s3_bucket}/{self.data_out_dir}"
        input_s3_uri = f"s3://{self.s3_bucket}/{self.data_inp_dir}/input.json"

        # create Hugging Face Model Class
        huggingface_model = HuggingFaceModel(name=self.model_name,
                                             model_data=self.model_uri,
                                             role=self.role, # iam role with permissions to create an Endpoint
                                             transformers_version=trans_version,
                                             pytorch_version=pytorch_version,
                                             py_version=py_version,
                                             sagemaker_session=self.sagemaker_session,
                                             env=self.env
                                            )

        # create a batch Transformer to run our batch job
        batch_job = huggingface_model.transformer(instance_count=self.instance_count,
                                                  instance_type=self.instance_type,
                                                  output_path=output_s3_path,
                                                  strategy='SingleRecord')

        # starts batch transform job and uses s3 data as input
        batch_job.transform(data=input_s3_uri,
                            content_type='application/json',
                            split_type='Line'
                            )

        batch_job.delete_model()
        self.logger.info(f"Batch Inference job has been completed successfully, and model {self.model_name} has been delete")