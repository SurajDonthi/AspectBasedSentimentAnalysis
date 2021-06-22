# Currently use Namespace variables, Need to add a hyperparameter tuning package
from argparse import Namespace

args = Namespace()

args.description = "Training with Restaurant Reviews data for Aspect Sentiment Analysis"
args.task = "aspect-sentiment"
args.log_path = "./logs"
args.max_epochs = 20
args.data_path = "./data/Restaurants_Train.xml"
args.test_path = "./data/restaurants-trial.xml"
args.dataset_args = dict()
args.encoder_args = dict(
    max_length=512,
    add_special_tokens=True,
    return_token_type_ids=False,
    return_attention_mask=True,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)
args.lr = 1e-3
args.train_split_ratio = 0.7
# args.limit_train_batches = 0.01
# args.limit_val_batches = 0.01
# args.limit_test_batches = 0.01
args.fast_dev_run = 10
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
