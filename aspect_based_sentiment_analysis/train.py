import os
from argparse import ArgumentParser
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (EarlyStopping, GPUStatsMonitor,
                                         LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import CometLogger, TestTubeLogger

from .pipeline import Pipeline
from .tuner import args as params
from .utils import save_args


def main(args):
    if not Path(args.log_path).exists():
        os.makedirs(args.log_path)
        version = 0
    else:
        version = int(os.listdir(args.log_path)[-1][-2:])
    save_dir = Path(args.log_path) / f'version_{version:02d}'
    loggers = [TestTubeLogger(
        save_dir=save_dir, name="",
        description=args.description, debug=args.debug,
        create_git_tag=args.git_tag, version=version,
        log_graph=True
    )]
    loggers[0].experiment
    loggers += [CometLogger(
        api_key='afAkCM1UJSi12AtcUYJPLdK9v',
        save_dir=save_dir,
        project_name='AspectBasedSentimentAnalysis',
        experiment_name='SentencePair' + f'_v{version}',
        offline=args.debug, version=version,
        log_graph=True
    )]

    checkpoint_dir = save_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor='Loss/val_loss',
        save_last=True,
        mode='min',
        save_top_k=10,
        period=5
    )
    early_stop_callback = EarlyStopping()

    model_pipeline = Pipeline.from_argparse_args(args)

    save_args(args, save_dir)

    trainer = Trainer.from_argparse_args(args,
                                         logger=loggers,
                                         callbacks=[early_stop_callback, chkpt_callback]
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

    main(args)
