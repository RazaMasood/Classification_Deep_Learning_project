import os
import gdown
import zipfile
from kidney_Disease_classifier import logger
from kidney_Disease_classifier.utils.common import get_size
from kidney_Disease_classifier.entity.config_entity import (DataIngestionConfig)

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        '''
        fetch data from source url
        '''

        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs('artifacts/data_ingestion', exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split('/')[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)

            logger.info(f'Download data from {dataset_url} into file {zip_download_dir}')

        except Exception as e:
            raise e
        
    def extract_zip_file(self):
        '''
        zip_file_path: str
        Extract the zip file into the data directory
        Function returns None
        '''
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_file:
            zip_file.extractall(unzip_path)