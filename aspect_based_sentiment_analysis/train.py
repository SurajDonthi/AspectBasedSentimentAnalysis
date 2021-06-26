import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

import comet_ml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, GPUStatsMonitor,
                                         LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import CometLogger, TestTubeLogger

from .pipeline import Pipeline
from .tuner import args as params
from .utils import save_args


def get_loggers_and_callbacks(args: Namespace, is_save_args: bool = True) -> None:

    version = get_version(args.log_path)
    save_dir = Path(args.log_path) / f'version_{version:02d}'
    os.makedirs(save_dir, exist_ok=True)

    if is_save_args:
        save_args(args, save_dir)

    loggers = [TestTubeLogger(
        save_dir=save_dir, name='',
        description=args.description, debug=args.debug,
        create_git_tag=args.git_tag, version=version,
        # log_graph=True
    )]
    loggers[0].experiment
    comet_log_path = save_dir / "comet_logs"
    os.makedirs(comet_log_path, exist_ok=True)
    loggers += [CometLogger(
        api_key='afAkCM1UJSi12AtcUYJPLdK9v',
        save_dir=comet_log_path,
        project_name='AspectBasedSentimentAnalysis',
        experiment_name='ABSA_SentencePair' + f'_v{version}',
        offline=args.debug,
        # log_graph=True
    )]

    checkpoint_dir = save_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='Loss/val_loss',
        save_last=True,
        mode='min',
        save_top_k=10,
        period=5
    )]
    callbacks += [EarlyStopping(
        monitor='Loss/val_loss',
        mode='min'
    )]
    callbacks += [GPUStatsMonitor(True, True, True, True)]
    callbacks += [LearningRateMonitor()]

    return loggers, callbacks


def get_version(log_path: Union[Path, str]) -> int:
    if not Path(log_path).exists():
        os.makedirs(log_path)
    version_folders = os.listdir(log_path)
    if version_folders:
        version = int(sorted(version_folders)[-1][-2:]) + 1
    else:
        version = 0
    return version


def train(args: Namespace):

    model_pipeline = Pipeline.from_argparse_args(args)

    loggers, callbacks = get_loggers_and_callbacks(args)

    trainer = Trainer.from_argparse_args(args,
                                         logger=loggers,
                                         callbacks=callbacks
                                         )

    trainer.fit(model_pipeline)
    trainer.test(model_pipeline)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser = Pipeline.add_model_specific_args(parser)
    parser = Pipeline.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if 'params' in locals():
        args.__dict__.update(params.__dict__)

    train(args)
