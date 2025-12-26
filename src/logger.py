import logging
import sys
from pathlib import Path

def setup_logger(name="StrokeSec", log_file=None):
    """
    Sets up a logger with console and file handlers.
    
    Args:
        name (str): Name of the logger.
        log_file (Path, optional): Path to the log file. 
                                   Defaults to 'logs/app.log' relative to project root.
    
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if function is called multiple times
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - [%(levelname)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file is None:
        # Determine project root (2 levels up from this file)
        project_root = Path(__file__).resolve().parent.parent
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "app.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# Create a default logger instance
logger = setup_logger()
