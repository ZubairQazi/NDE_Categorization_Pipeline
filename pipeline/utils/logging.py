# Logging utilities
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Args:
        name (str): The name of the logger, typically __name__
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handlers if the logger doesn't have any
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handler
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        # Set level to DEBUG
        logger.setLevel(logging.DEBUG)
    
    return logger