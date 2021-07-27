import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

import flat_table
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from transformers import (BertTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

from .base import BaseDataModule, BaseDataset
from .utils import (DirectoryNotFoundError, IsNotADirectoryError,
                    load_pretrained_model_or_tokenizer)


class SemEvalXMLDataset(Dataset):

    def __init__(
        self,
        filepath: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        encoder_args: Optional[dict] = dict(),
        # reader_args: Optional[dict] = dict(),
        **kwargs
    ):
        super().__init__()

        filepath = Path(filepath)
        self.df = self.load_sem_eval_data(filepath)
        self.preprocess_data()

        self.tokenizer = tokenizer
        self._encoder_args = encoder_args

    def preprocess_data(self):
        # filters
        not_null_entities = self.df['entities.term'].notnull()
        not_conflict = self.df['entities.polarity'] != 'conflict'
        self.df = self.df[not_null_entities & not_conflict].reset_index(drop=True)

        self.df['entities.polarity'] = pd.Categorical(
            self.df['entities.polarity'],
            categories=['negative', 'neutral', 'positive'])

        self.texts = self.df['text']
        self.aspects = self.df['entities.term']
        self.labels = self.df['entities.polarity'].cat.codes

    def load_xml_data_as_json(self, path: str):
        tree = ET.parse(path)
        root = tree.getroot()

        text_dataset = []
        for sentence in list(root):
            review = {}
            children = list(sentence)
            # print(children)
            text = children[0].text
            review['text'] = text

            if len(children) == 2:
                aspect_categories = children[1]

                entity_categories = []
                for category in list(aspect_categories):
                    entity_categories.append(category.attrib)

                review['entity_category'] = entity_categories

            elif len(children) == 3:
                aspect_terms = children[1]
                aspect_categories = children[2]

                entities = []
                for aspect_term in list(aspect_terms):
                    entities.append(aspect_term.attrib)
                review['entities'] = entities

                entity_categories = []
                for category in list(aspect_categories):
                    entity_categories.append(category.attrib)

                review['entity_category'] = entity_categories

            text_dataset.append(review)

        return text_dataset

    def load_json_as_df(self, json_data: dict):
        df = pd.read_json(json.dumps(json_data))
        df = flat_table.normalize(df).drop('index', axis=1)
        return df

    def load_sem_eval_data(self, path: str):
        return self.load_json_as_df(self.load_xml_data_as_json(path))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        target = self.labels[index]

        encodings = self.tokenizer.encode_plus(
            self.texts[index],
            self.aspects[index],
            **self._encoder_args
        )

        return {
            'input_ids': th.squeeze(encodings['input_ids'].type(th.long)),
            'attention_mask': th.squeeze(encodings['attention_mask']),
            'target': target
        }


class ReviewsMLMDataset(Dataset):

    def __init__(
        self,
        filepath: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        mask_pctg: float = .15,
        encoder_args: Optional[dict] = dict(
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding='max_length'
        ),
        read_args: Optional[dict] = dict()
    ):
        super().__init__()
        filepath = Path(filepath)

        self.data = pd.read_csv(filepath_or_buffer=filepath, **read_args)
        self.mask_pctg = mask_pctg
        self.tokenizer = tokenizer
        self._encoder_args = encoder_args

    def __len__(self):
        return len(self.data)

    def mask_inputs(self, inputs):
        rand_arr = th.rand_like(inputs.input_ids)
        mask_arr = (rand_arr < self.mask_pctg) * \
            (inputs.input_ids != 101) * (inputs.input_ids != 102)
        selection = th.flatten((mask_arr[0]).nonzero()).tolist()
        inputs.input_ids[0, selection] = 103
        return inputs

    def __getitem__(self, index):
        text = self.data.review_text[index]
        inputs = self.tokenizer.encode_plus(text=text, **self._encoder_args)
        labels = inputs.input_ids
        masked_inputs = self.mask_inputs(inputs)

        return masked_inputs, labels


class SemEvalDataModule(BaseDataModule):
    def __init__(
        self,
        train_path: str,
        test_path: Optional[str],
        train_val_split: float = 0.2,
        tokenizer: Optional[Union[PreTrainedTokenizer,
                                  PreTrainedTokenizerFast, str]] = 'bert-based-cased',
        encoder_args: Optional[dict] = dict(),
        train_bath_size: int = 16,
        val_bath_size: int = 16,
        test_bath_size: int = 16,
        # reader_args: Optional[dict] = dict(),
        **kwargs
    ):
        super().__init__()
        train_path = Path(train_path)
        test_path = Path(test_path)

        self.train_data = SemEvalXMLDataset(train_path, tokenizer, encoder_args)
        self.test_data = SemEvalXMLDataset(test_path, tokenizer, encoder_args)

        val_len = int(len(self.train_data) * train_val_split)
        train_len = len(self.train_data) - val_len
        self.train_data, self.val_data = random_split(self.train_data, [train_len, val_len])

        self.train_bath_size = train_bath_size
        self.val_bath_size = val_bath_size
        self.test_bath_size = test_bath_size

    def train_dataloader(self) -> DataLoader:
        pin_memory = th.has_cuda
        return DataLoader(self.train_data, self.train_batch_size,
                          shuffle=True, pin_memory=pin_memory,
                          num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        pin_memory = th.has_cuda
        return DataLoader(self.val_data, self.val_batch_size,
                          shuffle=True, pin_memory=pin_memory,
                          num_workers=self.num_workers)

    def test_dataloader(self) -> DataLoader:
        pin_memory = th.has_cuda
        return DataLoader(self.test_data, self.test_batch_size,
                          shuffle=True, pin_memory=pin_memory,
                          num_workers=self.num_workers)
