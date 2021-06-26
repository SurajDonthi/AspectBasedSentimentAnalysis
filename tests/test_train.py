from aspect_based_sentiment_analysis.train import train
from aspect_based_sentiment_analysis.tuner import args


def test_train():
    train(args)
