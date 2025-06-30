"""Helper method for logging."""

import datetime
import logging
from pathlib import Path
import sys


def setup_logger(logger: logging.Logger, save_dir: Path) -> None:
    """Preconfigures a given logger.

    Parameters
    ----------
    logger
        logger to preconfigure
    save_dir
        directory for pipeline to save model artifacts too. Log outputs to
        log/model_name/version_datetime dir
    """
    logger.setLevel(logging.INFO)
    log_file = save_dir / Path("run.log")

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("%(levelname)s %(name)s %(asctime)s %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def create_save_dir(model_name: str) -> Path:
    """Create a directory to save log and model artifacts for an experiment.

    Parameters
    ----------
    model_name
        name of model, the subdir to save files

    Returns
    -------
    save_dir
        directory for pipeline to save model artifacts to. Log outputs to
        log/model_name/version_datetime dir
    """
    curr_dir = Path.cwd()
    model_dir = Path(model_name).parts[-1]
    experiment_dir = Path("version_" + str(datetime.datetime.now()))

    save_dir = curr_dir / "log" / model_dir / experiment_dir

    if not save_dir.is_dir():
        save_dir.mkdir()

    return save_dir
