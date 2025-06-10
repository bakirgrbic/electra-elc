"""Helper methods for logging"""

from pathlib import Path
import logging
import datetime


def get_my_logger(model_name, task_name):
    """Prefigures a logging.Logger object to organize log files in log/.

    Keyword Arguments:
    model_name -- name of model experiment, also the subdir to collect log files
    task_name -- name of the log file to differentiate between tasks

    Returns:
    logging.Logger object prefigured to save information to log file in 
    log/model_name directory.
    """
    curr_dir = Path(Path.cwd())
    target_dir = curr_dir / "log" / Path(model_name).parts[-1]

    if not target_dir.is_dir():
        target_dir.mkdir()

    logger = logging.getLogger(task_name)
    log_file = target_dir / Path(f"{task_name}-{datetime.datetime.now()}.log")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    log(logger, f"Using {model_name} to perform {task_name}")
    return logger


def log(logger, message, message_type='info'):
    """Logs a message to appropriate type given a logging.Logger object"""
    print(message)
    if message_type == 'info':
        logger.info(message)
    elif message_type == 'warning':
        logger.warning(message)
    elif message_type == 'error':
        logger.error(message)
    elif message_type == 'critical':
        logger.critical(message)
