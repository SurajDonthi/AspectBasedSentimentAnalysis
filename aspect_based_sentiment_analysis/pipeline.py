import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, Optional, Union

import torch as th
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import (
    multiclass_auroc, precision, precision_recall, recall)
from pytorch_lightning.metrics.functional.confusion_matrix import \
    confusion_matrix
from pytorch_lightning.utilities import parsing
from torch.utils.data import DataLoader, random_split
from transformers import (AdamW, BertModel, BertTokenizerFast, DistilBertModel,
                          DistilBertTokenizerFast, PreTrainedTokenizer,
                          PreTrainedTokenizerFast, SqueezeBertModel,
                          SqueezeBertTokenizerFast,
                          get_linear_schedule_with_warmup)
from typing_extensions import Literal

from .base import BaseModule
from .data import SemEvalXMLDataset
from .models import (DummyClassifier, SequenceClassifierModel,
                     TokenClassifierModel)
from .utils import load_pretrained_model_or_tokenizer

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler(sys.stdout)
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)

LOSSES = {'bce': F.binary_cross_entropy,
          'bce_logits': F.binary_cross_entropy_with_logits,
          'cross_entropy': F.cross_entropy, 'nll_loss': F.nll_loss,
          'kl_div': F.kl_div, 'mse': F.mse_loss,
          'l1_loss': F.l1_loss}

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

# SCHEDULERS = {
#     'linear_with_warmup': get_linear_schedule_with_warmup
# }


class Pipeline(BaseModule):

    def __init__(
        self,
        data_path: str,
        test_path: Optional[str] = None,
        bert_base: str = 'bert',
        task: Literal[tuple(TASKS.keys())] = 'classification',
        tokenizer: Optional[Union[PreTrainedTokenizer, PreTrainedTokenizerFast]] = None,
        train_split_ratio: float = 0.7,
        train_batchsize: int = 32,
        val_batchsize: int = 32,
        test_batchsize: int = 32,
        num_workers: int = 4,
        lr: float = 5e-5,
        criterion: Literal[tuple(LOSSES.keys())] = 'cross_entropy',
        freeze_bert: bool = False,
        dataset_args: dict = dict(),
        encoder_args: dict = dict(),
        model_args: dict = dict(dropout=0.3),
        optim_args: dict = dict(eps=1e-8),
        #  scheduler_args: dict = dict(),
        * args, **kwargs
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.test_path = Path(test_path)

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"File '{self.data_path.absolute().as_posix()}' does not exist!")
        if not self.test_path.exists():
            raise FileNotFoundError(
                f"File '{self.data_path.absolute().as_posix()}' does not exist!")

        self.train_split_ratio = train_split_ratio
        self.train_batchsize = train_batchsize
        self.test_batchsize = test_batchsize
        self.val_batchsize = val_batchsize
        self.num_workers = num_workers
        dataset_args.update(dict(encoder_args=encoder_args))
        self._dataset_args = dataset_args

        self.criterion = LOSSES[criterion]
        self.lr = lr
        self.optim_args = optim_args

        self.freeze_bert = freeze_bert
        bert_args = BERT_BASE[bert_base]

        pretrained_model_name = Path(bert_args['pretrained_model_name'])
        self.bert_base = load_pretrained_model_or_tokenizer(
            bert_args['model'], (pretrained_model_name))

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = load_pretrained_model_or_tokenizer(
                bert_args['tokenizer'], pretrained_model_name)

        task_args = TASKS[task]
        self.classifier = task_args['model'](**model_args)
        self._dataset = task_args['dataset']

        self.save_hyperparameters()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--description', required=False, type=str)
        parser.add_argument('--log_path', type=str,
                            default='./.logs')
        parser.add_argument('--git_tag', required=False, const=False,
                            type=parsing.str_to_bool, nargs='?')
        parser.add_argument('--debug', required=False, const=False,
                            nargs='?', type=parsing.str_to_bool)
        return parser

    def prepare_data(self):
        self.train_data = self._dataset(self.data_path, self.tokenizer, **self._dataset_args)
        # self.train_data.tokenizer = self.tokenizer
        len_ = len(self.train_data)
        train_len = int(len_ * self.train_split_ratio)
        val_len = len_ - train_len
        print(f'Train length: {train_len}, Val length: {val_len}')

        self.train_data, self.val_data = random_split(self.train_data, [train_len, val_len])
        # self.logger.experiment[0].log_graph(self.train_data[0])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batchsize,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if self.val_data:
            return DataLoader(self.val_data, batch_size=self.train_batchsize,
                              shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        if not self.test_path:
            return self.val_dataloader()
        self.test_data = self._dataset(self.test_path, self.tokenizer, **self._dataset_args)
        return DataLoader(self.test_data, batch_size=self.train_batchsize,
                          shuffle=False, num_workers=self.num_workers)

    def configure_optimizers(self):
        params = []
        if not self.hparams.freeze_bert:
            params += list(self.bert_base.parameters())
        params += list(self.classifier.parameters())

        optim = AdamW(params,
                      lr=self.hparams.lr, **self.optim_args)
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=200,
            num_training_steps=self.num_training_steps
        )
        return [optim], [scheduler]

    def forward(self, batch):
        embeddings = self.bert_base(**batch)
        return self.classifier(embeddings)

    def shared_step(self, batch, return_preds=False):
        targets = batch.pop('target').type(th.long)
        out = self(batch)

        loss = self.criterion(out, targets)
        _, preds = th.max(out, dim=1)
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
        self.log_dict(logs, on_epoch=True, on_step=True)

        pbar_logs = logs = {
            'loss_train': loss,
            'train_acc': acc
        }
        self.log_dict(pbar_logs, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch:
            loss, acc = self.shared_step(batch)

            logs = {
                'Loss/val_loss': loss,
                'Accuracy/val_acc': acc,
            }
            self.log_dict(logs)
            self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        targets = batch['target']
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
        self.log('test_acc', acc, prog_bar=True)

    def get_progress_bar_dict(self) -> Dict[str, Union[int, str]]:
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
