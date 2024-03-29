import pandas as pd
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

import logging
logging.basicConfig(level=logging.INFO)

# This parameter is important for running the torch on MacOS.
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"


class Dataset:

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class BatchEncoder:
    """
    Encoding dataset with sentence transformers in batches
    """
    def __init__(self, dataframe: pd.DataFrame, column_to_encode: str,
                 embedding_column_name: str = "embedding", batch_size: int = 64):
        self.df = dataframe
        self.column = column_to_encode
        self.embedding_column = embedding_column_name
        self.batch_size = batch_size
        self.model_name = "baai/bge-large-en-v1.5"
        self.model = SentenceTransformer(self.model_name)

        logging.info(
            f'Batch Encoder for column {self.column} '
            f'initialized with batch size {self.batch_size}.')

    def create_dataset_and_dataloader(self) -> DataLoader:
        logging.info('Creating dataset...')
        dataset = Dataset(self.df[self.column].tolist())
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        logging.info('Dataset created.')
        return dataloader

    def encode_dataset(self, dataloader: DataLoader) -> list:
        logging.info('Starting encoding...')
        embeddings = []
        for i, batch in tqdm(enumerate(dataloader), desc="Encoding"):
            embedding = self.model.encode(batch)
            embeddings.extend(embedding.tolist())
        return embeddings

    def assign_embeddings_to_dataframe(self, embeddings: list):
        logging.info('Assigning embeddings to new dataframe column "embeddings".')
        self.df[self.embedding_column] = embeddings

    def process_data(self) -> pd.DataFrame:
        dataloader = self.create_dataset_and_dataloader()
        embeddings = self.encode_dataset(dataloader)
        self.assign_embeddings_to_dataframe(embeddings)
        logging.info('Data processing complete.')
        return self.df
