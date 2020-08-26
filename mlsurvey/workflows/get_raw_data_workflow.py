import os

import boto3
import luigi

from mlsurvey.workflows.tasks.base_task import BaseTask


class GetRawDataWorkflow(BaseTask):
    """
    Workflow to get the raw data from external source. S3 source at the moment
    """

    def run(self):
        """
        Open a s3 session, download files
        """
        source_path = self.config.data['file']['source']['path']
        source_name = self.config.data['file']['source']['filename']
        base_url = self.config.data['base_url']
        bucket_name = self.config.data['bucket_name']

        s3_session = boto3.session.Session(profile_name=str(base_url))
        s3_client = s3_session.client('s3', endpoint_url=str(base_url))

        complete_source_path = os.path.join(str(source_path), str(source_name))

        # Télécharger des fichiers
        with open(self.output().path, 'wb') as f:
            s3_client.download_fileobj(bucket_name, complete_source_path, f)

    def output(self):
        destination_path = self.config.data['file']['destination']['path']
        destination_name = self.config.data['file']['destination']['filename']
        complete_destination_path = os.path.join(str(self.base_directory), str(destination_path), str(destination_name))
        target = luigi.LocalTarget(complete_destination_path)
        target.makedirs()
        return target
