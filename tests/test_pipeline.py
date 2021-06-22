from argparse import ArgumentParser

from aspect_based_sentiment_analysis.pipeline import Pipeline
from aspect_based_sentiment_analysis.train import main
from aspect_based_sentiment_analysis.tuner import args
from pytorch_lightning import Trainer


def test_sanity():
    main(args)
