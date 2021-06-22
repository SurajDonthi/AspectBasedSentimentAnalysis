from pathlib import Path

import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset, random_split

from base import BaseDataset


class SentimentDataset(BaseDataset):

    def __init__(self, file_path: str,
                 max_len=512,
                 read_args: dict = dict(usecols=['review_body', 'sentiment']),
                 **kwargs
                 ):
        self.df = pd.read_csv(file_path, **read_args)
        input, labels = self.df.iloc[:, 0], self.df.iloc[:, 1]
        super().__init__(input, labels, max_len)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return super().__getitem__(index)
