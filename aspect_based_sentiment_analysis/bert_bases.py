from transformers import (BertModel, BertTokenizerFast, DistilBertModel,
                          DistilBertTokenizerFast, SqueezeBertModel,
                          SqueezeBertTokenizerFast)

BERT_BASE = {
    'bert': {'model': BertModel,
             'tokenizer': BertTokenizerFast,
             'pretrained_model_name': '.weights/bert-base-cased',
             },
    'distil': {'model': DistilBertModel,
               'tokenizer': DistilBertTokenizerFast,
               'pretrained_model_name': '.weights/distilbert-base-multilingual-cased'
               },
    'squeeze': {'model': SqueezeBertModel,
                'tokenizer': SqueezeBertTokenizerFast,
                'pretrained_model_name': '.weights/squeezebert/squeezebert-uncased'
                }
}
