import csv
import inspect
from argparse import Namespace
from pathlib import Path

from loguru import logger


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


def load_pretrained_model_or_tokenizer(model_or_tokenizer, model_path, download=True):
    if model_path.exists():
        try:
            pretrained_model_or_tokenizer = model_or_tokenizer.from_pretrained(
                model_path.absolute())
            return pretrained_model_or_tokenizer
        except Exception:
            logger.info('Model does not exist. Downloading the model...')

    if len(model_path.parts) > 2:
        path = model_path.parts[-2] + '/' + model_path.name
    else:
        path = model_path.name

    pretrained_model_or_tokenizer = model_or_tokenizer.from_pretrained(path)
    if download:
        pretrained_model_or_tokenizer.save_pretrained(model_path.absolute())

    return pretrained_model_or_tokenizer


class DirectoryNotFoundError(OSError):
    """
    Folder not found.
    """


class IsNotADirectoryError(OSError):
    "Not a folder."
