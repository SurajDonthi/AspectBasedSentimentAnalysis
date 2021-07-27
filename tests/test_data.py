import logging

import pytest
from aspect_based_sentiment_analysis.data import (ReviewsMLMDataset,
                                                  SemEvalXMLDataset)
from transformers import BertTokenizerFast


@pytest.fixture()
def semeval_dataset():
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    encoder_args = dict(
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    dataset = SemEvalXMLDataset('./data/raw/restaurants-trial.xml',
                                tokenizer, encoder_args)
    return dataset


def test_xml_dataset_init(semeval_dataset):
    assert len(semeval_dataset) == 130


def test_xml_data_getitem(semeval_dataset):
    inputs = semeval_dataset[0]
    assert isinstance(inputs, dict)
    assert 'input_ids' in inputs
    assert 'attention_mask' in inputs


@pytest.fixture()
def reviews_dataset():
    filepath = 'data/raw/reviews_data.csv'
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

    dataset = ReviewsMLMDataset(filepath, tokenizer,
                                read_args=dict(usecols=['review_text']))

    return dataset


def test_reviews_mlm_dataset_init(reviews_dataset):
    assert len(reviews_dataset) == 143443


def test_reviews_mlm_dataset_getitem(semeval_dataset):
    inputs = semeval_dataset[0]
    assert isinstance(inputs, dict)
    assert 'input_ids' in inputs
    assert 'attention_mask' in inputs
