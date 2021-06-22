from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import (
    f1_score, multiclass_auroc, precision, precision_recall, recall)
from pytorch_lightning.metrics.functional.confusion_matrix import \
    confusion_matrix
from pytorch_lightning.utilities import parsing
from torch.utils.data import DataLoader, random_split
from transformers import (AdamW, BertModel, BertTokenizer, DistilBertModel,
                          DistilBertTokenizer, PreTrainedTokenizer,
                          SqueezeBertModel, SqueezeBertTokenizer,
                          get_linear_schedule_with_warmup)
from typing_extensions import Literal

from base import BaseModule
from data import SentimentDataset
from models import (DummyClassifier, SequenceClassifierModel,
                    TokenClassifierModel)

LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss}

BERT_BASE = {
    'bert': {'model': BertModel,
             'tokenizer': BertTokenizer,
             'pretrained_model_name': 'bert-base-cased',
             },
    'distil': {'model': DistilBertModel,
               'tokenizer': DistilBertTokenizer,
               'pretrained_model_name': 'distilbert-base-multilingual-cased'
               },
    'squeeze': {'model': SqueezeBertModel,
                'tokenizer': SqueezeBertTokenizer,
                'pretrained_model_name': 'squeezebert/squeezebert-uncased'
                }
}

TASKS = {
    'classification': {
        'model': SequenceClassifierModel,
        'dataset': None
    },
    'sentiment-analysis': {
        'model': SequenceClassifierModel,
        'dataset': SentimentDataset
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


class Pipeline(BaseModule):

    def __init__(self,
                 data_path: str,
                 bert_base: str = 'bert',
                 task: Literal[tuple(TASKS.keys())] = 'classification',
                 tokenizer: Optional[Union[PreTrainedTokenizer]] = None,
                 train_split_ratio: float = 0.7,
                 train_batchsize: int = 32,
                 val_batchsize: int = 32,
                 test_batchsize: int = 32,
                 num_workers: int = 4,
                 lr: float = 5e-5,
                 criterion: Literal[tuple(LOSSES.keys())] = 'cross_entropy',
                 freeze_bert: bool = False,
                 data_args: dict = dict(max_len=512,
                                        read_args=dict(nrows=3500,
                                                       usecols=['review_body', 'sentiment'])
                                        ),
                 model_args: dict = dict(dropout=0.3),
                 optim_args: dict = dict(eps=1e-8),
                 *args, **kwargs):
        super().__init__()

        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise Exception(
                f"Path '{self.data_path.absolute().as_posix()}' does not exist!")

        self.train_split_ratio = train_split_ratio
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.val_batchsize = val_batchsize
        self.num_workers = num_workers
        self._data_args = data_args

        self.criterion = LOSSES[criterion]
        self.lr = lr
        self.optim_args = optim_args

        self.freeze_bert = freeze_bert
        bert_args = BERT_BASE[bert_base]
        self.pretrained_model_name = bert_args['pretrained_model_name']
        self.bert_base = bert_args['model'].from_pretrained(self.pretrained_model_name)
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = bert_args['tokenizer']\
                .from_pretrained(self.pretrained_model_name)

        task_args = TASKS[task]
        self.classifier = task_args['model'](**model_args)
        self.Dataset = task_args['dataset']

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('-des', '--description', required=False, type=str)
        parser.add_argument('-lp', '--log_path', type=str,
                            default='./logs')
        parser.add_argument('-gt', '--git_tag', required=False, const=False,
                            type=parsing.str_to_bool, nargs='?')
        parser.add_argument('--debug', required=False, const=False, nargs='?',
                            type=parsing.str_to_bool)
        return parser

    def prepare_data(self):
        self.train_data = self.Dataset(self.data_path, **self._data_args)
        self.train_data._tokenizer = self.tokenizer
        len_ = len(self.train_data)
        train_len = int(len_ * self.train_split_ratio)
        val_len = len_ - train_len
        print(f'Train length: {train_len}, Val length: {val_len}')

        self.train_data, self.val_data = random_split(self.train_data, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_data:
            loader = DataLoader(self.val_data, batch_size=self.train_batchsize,
                                shuffle=True, num_workers=self.num_workers)
            return loader

    def test_dataloader(self):
        return self.val_dataloader()

    def configure_optimizers(self):
        params = []
        if self.hparams.freeze_bert:
            params += list(self.bert_base.parameters())
        params += list(self.classifier.parameters())

        optim = AdamW(params,
                      lr=self.hparams.lr, **self.optim_args)
        # scheduler = get_linear_schedule_with_warmup(optim,
        #                                             num_warmup_steps=0,  # Default value
        #                                             num_training_steps=self.trainer.max_epochs
        #                                             )
        return [optim] \
            # , [scheduler]

    def forward(self, batch):
        input_ids, attention_mask = batch

        embeddings = self.bert_base(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    )
        return self.classifier(embeddings)

    def shared_step(self, batch, return_preds=False):
        *batch, targets = batch
        out = self(batch)

        loss = self.criterion(out, targets)
        _, preds = torch.max(out, dim=1)
        acc = (preds == targets).float().mean()

        if return_preds:
            return loss, acc, out, preds
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.shared_step(batch)

        logs = {
            'Loss/train_loss': loss,
            'Accuracy/train_acc': acc
        }
        self.log_dict(logs, prog_bar=True, on_epoch=True, on_step=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch:
            loss, acc = self.shared_step(batch)

            logs = {
                'Loss/val_loss': loss,
                'Accuracy/val_acc': acc,
                'learning_rate': self.trainer.lightning_optimizers[0].param_groups[0]['lr']
            }
            self.log_dict(logs, prog_bar=True)

    def test_step(self, batch, batch_idx):
        *_, targets = batch
        loss, acc, out, preds = self.shared_step(batch, return_preds=True)

        # p = precision(out, targets)
        # auroc = multiclass_auroc(out, targets)

        logs = {
            'Loss/test_loss': loss,
            'Accuracy/test_acc': acc,
            # 'Precision': p,
            # 'Multiclass AUROC': auroc
        }
        self.log_dict(logs)
