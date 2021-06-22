# Currently use Namespace variables, Need to add a hyperparameter tuning package
from argparse import Namespace

args = Namespace()

args.description = "Training with Amazon outdoors data for Sentiment Analysis"
args.task = "sentiment-analysis"
args.log_path = "./logs"
args.max_epochs = 20
args.data_path = "./data/amazon_outdoors/amazon_reviews_us_Outdoors_v1_00_rebalanced.csv.gz"
args.data_args = dict(max_len=512,
                      read_args=dict(nrows=7500,
                                     usecols=['review_body', 'sentiment'])
                      )
args.lr = 1e-3
args.train_split_ratio = 0.7
# args.limit_train_batches = 0.01
# args.limit_val_batches = 0.01
# args.limit_test_batches = 0.01
args.train_batchsize = 16
args.val_batchsize = 16
args.test_batchsize = 16
args.debug = True
args.git_tag = True
args.gradient_clip_val = 1.0

args.gpus = 1
args.weights_summary = 'full'
args.profiler = True
# args.early_stop_callback = False
