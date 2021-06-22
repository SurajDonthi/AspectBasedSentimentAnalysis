import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import flat_table
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, random_split
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .base import BaseDataset


class SemEvalXMLDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        encoder_args={},
        reader_args: dict = dict(),
        **kwargs
    ):
        super().__init__()

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist!!")
        self.df = self.load_sem_eval_data(file_path)

        # filters
        not_null_entities = self.df['entities.term'].notnull()
        not_conflict = self.df['entities.polarity'] != 'conflict'
        self.df = self.df[not_null_entities & not_conflict]

        self.df['entities.polarity'] = pd.Categorical(self.df['entities.polarity'], categories=[
            'negative', 'neutral', 'positive', 'conflict'])

        self.texts = self.df['text']
        self.aspects = self.df['entities.term']
        self.labels = self.df['entities.polarity'].codes

    def load_xml_data_as_json(self, path: str):
        tree = ET.parse(path)
        root = tree.getroot()

        text_dataset = []
        for sentence in root.getchildren():
            review = {}
            children = sentence.getchildren()
            # print(children)
            text = children[0].text
            review['text'] = text

            if len(children) == 2:
                aspect_categories = children[1]

                entity_categories = []
                for category in aspect_categories.getchildren():
                    entity_categories.append(category.attrib)

                review['entity_category'] = entity_categories

            elif len(children) == 3:
                aspect_terms = children[1]
                aspect_categories = children[2]

                entities = []
                for aspect_term in aspect_terms.getchildren():
                    entities.append(aspect_term.attrib)
                review['entities'] = entities

                entity_categories = []
                for category in aspect_categories.getchildren():
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
            max_length=512,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        return {
            'input_ids': th.squeeze(th.tensor(encodings['input_ids'], dtype=th.long)),
            'attention_mask': th.squeeze(th.tensor(encodings['attention_mask'])),
            'target':}
