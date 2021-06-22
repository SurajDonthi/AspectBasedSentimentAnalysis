import csv
import inspect
from argparse import Namespace
from pathlib import Path


def save_args(args: Namespace, save_dir: Path) -> None:
    with open(save_dir / 'hparams.csv', 'w') as f:
        csvw = csv.writer(f)
        csvw.writerow(['hparam', 'value'])
        for k, v in args.__dict__.items():
            csvw.writerow([k, v])


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def update_args(func, args):
    default_args = get_default_args(func)

    args = {k: v for k, v in args.items() if k in default_args.keys()}

    default_args.update(args)

    return default_args
