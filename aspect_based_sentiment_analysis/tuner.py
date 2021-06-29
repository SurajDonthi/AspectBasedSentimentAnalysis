# Currently use Namespace variables,
# TODO: Need to add a hyperparameter tuning package
# Possibly change from namespace to something else like Hydra, OmegaConf
from argparse import Namespace

import optuna

# from .train import train


# def objective(trial: optuna.trial.Trial) -> float:

#     return train(args)


def tuner():
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=20)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))


args = Namespace()

args.description = "Training with Restaurant Reviews data for Aspect Sentiment Analysis. With Kaiming Normal Initialization"
args.task = "aspect-sentiment"
args.log_path = "./.logs"
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
# args.fast_dev_run = 2
args.train_batchsize = 16
args.val_batchsize = 16
args.test_batchsize = 16
args.num_workers = 12
args.debug = False
args.git_tag = True
args.gradient_clip_val = 1.0

args.gpus = 1
# args.log_gpu_memory = 'all'
args.weights_summary = 'top'
args.profiler = 'simple'
# args.early_stop_callback = False
