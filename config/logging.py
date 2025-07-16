import logging
import os
from pathlib import Path


def setup_logging(level: str = "INFO", log_to_file: bool = False, log_file: str = None):
    """
    Setup unified logging configuration for all Niffler scripts.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to also log to file
        log_file: Optional specific log file path
    """
    # Create logs directory if logging to file
    if log_to_file and not log_file:
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        log_file = logs_dir / "niffler.log"
    
    # Configure basic logging
    if log_to_file and log_file:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )