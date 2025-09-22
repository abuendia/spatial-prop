import sys
import os
from datetime import datetime


def setup_logging_to_file(save_dir: str):
    """Set up logging to redirect stdout and stderr to a file in the save directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"out_{timestamp}.log")
    
    # Open log file
    log_file_handle = open(log_file, 'w')
    
    # Redirect stdout and stderr to file only
    sys.stdout = log_file_handle
    sys.stderr = log_file_handle
    
    # Write initial message to log file
    print(f"Logging setup complete. All output will be saved to: {log_file}")
    return log_file
