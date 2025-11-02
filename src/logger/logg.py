import logging 
import os

def logs(file_name: str, log_dir: str = r"D:\medicare\src\logger\logs"):
    """
    Create a logger that writes to console and a file in a specific directory.

    Args:
        file_name (str): Name of the log file (e.g., 'app.log').
        log_dir (str): Directory where logs should be saved. Defaults to 'logs'.
    """
    # Ensure the directory exists
    os.makedirs(log_dir, exist_ok=True)

    # Full path to the log file
    file_path = os.path.join(log_dir, file_name)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(file_path, mode="a", encoding="utf-8")

        formatter = logging.Formatter(
            "{asctime} - {levelname} - {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M"
        )

        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger