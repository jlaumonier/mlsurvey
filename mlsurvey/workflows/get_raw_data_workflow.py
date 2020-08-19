import luigi
import boto3
import os


class GetRawDataWorkflow(luigi.Task):
    """
    Workflow to get the raw data from external source. S3 source at the moment
    """
    destination_path = luigi.Parameter()
    destination_name = luigi.Parameter()
    source_path = luigi.Parameter()
    source_name = luigi.Parameter()
    base_url = luigi.Parameter()
    bucket_name = luigi.Parameter()

    def run(self):
        """
        Open a s3 session, download files
        """
        s3_session = boto3.session.Session(profile_name=str(self.base_url))
        s3_client = s3_session.client('s3', endpoint_url=str(self.base_url))

        complete_destination_path = os.path.join(str(self.destination_path), str(self.destination_name))
        complete_source_path = os.path.join(str(self.source_path), str(self.source_name))

        # Télécharger des fichiers
        with open(complete_destination_path, 'wb') as f:
            s3_client.download_fileobj(self.bucket_name, complete_source_path, f)

    def output(self):
        complete_path = os.path.join(str(self.destination_path), str(self.destination_name))
        target = luigi.LocalTarget(complete_path)
        target.makedirs()
        return target
