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
    loggers = [TestTubeLogger(save_dir=args.log_path, name="",
                              description=args.description, debug=args.debug,
                              create_git_tag=args.git_tag)]
    loggers[0].experiment
    log_dir = Path(loggers[0].save_dir) / f"version_{loggers[0].version}"
    loggers += [CometLogger(api_key='afAkCM1UJSi12AtcUYJPLdK9v',
                            project_name='AspectBasedSentimentAnalysis',
                            experiment_name='SentencePair' + f'_v{loggers[0].version}',
                            offline=args.debug)]

    checkpoint_dir = log_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    chkpt_callback = ModelCheckpoint(checkpoint_dir,
                                     monitor='Loss/val_loss',
                                     save_last=True,
                                     mode='min',
                                     save_top_k=10,
                                     period=5
                                     )
    early_stop_callback = EarlyStopping()

    model_pipeline = Pipeline.from_argparse_args(args)

    save_args(args, log_dir)

    trainer = Trainer.from_argparse_args(args, loggers=loggers,
                                         checkpoint_callback=chkpt_callback, 
                                         early_stop_callback=early_stop_callback
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
