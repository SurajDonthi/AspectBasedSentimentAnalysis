from argparse import ArgumentParser

from aspect_based_sentiment_analysis.pipeline import Pipeline
from aspect_based_sentiment_analysis.train import main
from aspect_based_sentiment_analysis.tuner import args as params
from pytorch_lightning import Trainer


def test_sanity():
    parser = ArgumentParser()
    parser = Pipeline.add_model_specific_args(parser)
    parser = Pipeline.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    if 'params' in locals():
        args.__dict__.update(params.__dict__)

    main(args)
