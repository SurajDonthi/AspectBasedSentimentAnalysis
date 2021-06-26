import pytest
from aspect_based_sentiment_analysis.data import SemEvalXMLDataset
from transformers import BertTokenizerFast


@pytest.fixture()
def semeval_dataset():
    tokenizer = BertTokenizerFast.from_pretrained('.weights/bert-base-cased')
    encoder_args = dict(
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    dataset = SemEvalXMLDataset('./data/restaurants-trial.xml',
                                tokenizer, encoder_args)
    return dataset


def test_xml_dataset_init(dataset):
    assert len(dataset) == 130


def test_xml_data_getitem(semeval_dataset):
    inputs = semeval_dataset[0]
    assert isinstance(inputs, dict)
    assert 'input_ids' in inputs
    assert 'attention_mask' in inputs
