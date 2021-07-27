from .datasets import SemEvalXMLDataset
from .models import (DummyClassifier, SequenceClassifierModel,
                     TokenClassifierModel)

TASKS = {
    'classification': {
        'model': SequenceClassifierModel,
        'dataset': None
    },
    # 'sentiment-analysis': {
    #     'model': SequenceClassifierModel,
    #     'dataset': SentimentDataset
    # },
    'aspect-sentiment': {
        'model': SequenceClassifierModel,
        'dataset': SemEvalXMLDataset
    },
    'ner': {
        'model': TokenClassifierModel,
        'dataset': None
    },
    'pos-tagging': {
        'model': TokenClassifierModel,
        'dataset': None
    },
    'semantic-similarity': {
        'model': DummyClassifier,
        'dataset': None
    },
}
