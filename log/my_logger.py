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
    curr_dir = Path(Path.cwd().parts[-1])
    relative_log_dir = curr_dir.relative_to("log", walk_up=True)
    target_dir = relative_log_dir / Path(model_name)

    if not target_dir.is_dir():
        target_dir.mkdir()

    logger = logging.getLogger(task_name)
    log_file = target_dir / Path(f"{task_name}-{datetime.datetime.now()}.log")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler) 
    logger.setLevel(logging.INFO)

    log(logger, "Using {model_name} to perform {task_name}")
    return logger

def log(logger, message, type='info'):
    if type == 'info':
        logger.info(message)
    elif type == 'warning':
        logger.warning(message)
    elif type == 'error':
        logger.error(message)
    elif type == 'critical':
        logger.critical(message)
