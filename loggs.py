# loggs.py
import logging  # Import the logging module to enable logging within the application
from config import paths  # Import paths from config.py

def setup_logger(log_file='process.log', log_level=logging.INFO):
    """
    Sets up a logger with both file and console handlers.

    Parameters:
    log_file (str): The name of the log file where logs will be saved. Defaults to 'process.log'.
    log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.

    Returns:
    logger (Logger): Configured logger instance.
    """
    # Create a logger instance with the name of the current module
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    
    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create a console handler to log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create a formatter to specify the format of the log messages
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)  # Apply the formatter to the file handler
    console_handler.setFormatter(formatter)  # Apply the formatter to the console handler
    
    # Add both handlers (file and console) to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger  # Return the configured logger

# Create a global logger instance using the setup_logger function
logger = setup_logger()