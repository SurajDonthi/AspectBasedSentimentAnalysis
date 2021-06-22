import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Union

import flat_table
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, random_split
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .base import BaseDataset


class SemEvalXMLDataset(BaseDataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        encoder_args={},
        reader_args: dict = dict(),
        **kwargs
    ):
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist!!")
        df = self.load_xml_data()  # To Implement
        input, labels = self.df.iloc[:, 0], self.df.iloc[:, 1]
        super().__init__(input, labels, tokenizer, encoder_args)

    def load_xml_data(self, path):
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

    def load_json_as_df(self, json_data):
        df = pd.read_json(json.dumps(json_data))
        df = flat_table.normalize(df).drop('index', axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        target = self.labels[index]

        encodings = self.tokenizer.encode_plus(
            self.text[index],
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
            'target': }
