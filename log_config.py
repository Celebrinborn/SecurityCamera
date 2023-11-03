import logging
import logging.config
import os
import yaml
from typing import Optional
from pathlib import Path
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = record.__dict__.copy()
        if record.exc_info:
            log_record['exc_info'] = self.formatException(record.exc_info)
        if record.stack_info:
            log_record['stack_info'] = self.formatStack(record.stack_info)
        return json.dumps(log_record)

def configure_logging(log_file_name: Optional[str] = None, *args, clear_log_file: bool = False):
    """Configure logging based on the given log file name.

    If the log file name is not given, check the LOG_FILE_NAME environment
    variable. If that is not set, use the default file name `app.log`.

    Args:
        log_file_name (str): The name of the log file to use.

    Raises:
        FileNotFoundError: If the logging configuration file is not found.
    """
    # Set the log file name using the provided argument or an environment variable
    log_file_name = log_file_name if log_file_name is not None else os.environ.get('LOG_FILE_NAME', 'app.log')

    # If the log file should be cleared, delete it and print a message to the user
    if clear_log_file:
        Path('logs', log_file_name).unlink(missing_ok=True) # type: ignore

    # Check if logging configuration file exists
    if not os.path.exists('logging.yaml'):
        # If it doesn't exist, print a message to the user and crash
        print("Logging configuration file not found.")
        print("Current working directory: ", os.getcwd())
        print("Files and directories in current directory: ", os.listdir())
        print("All files and directories in subdirectories: ")
        for root, dirs, files in os.walk("."):
            for name in files + dirs:
                print(os.path.join(root, name))
        raise FileNotFoundError("logging.yaml file not found.")

    # Configure the logging module
    with open('logging.yaml', 'r') as f:
        config = yaml.safe_load(f.read())
        # Create the logs directory if it doesn't exist
        Path('logs').mkdir(parents=True, exist_ok=True)
        # Update the filename in the configuration dictionary
        config['handlers']['file']['filename'] = str(Path('logs', log_file_name)) # type: ignore
        logging.config.dictConfig(config)


import inspect
import json
import numpy as np
import sys

def DumpLocalVarsAsJson():
    # Get the calling frame (the frame of the function that called this function)
    calling_frame = inspect.currentframe().f_back
    
    # Create an empty dictionary to store the information for each local variable
    local_vars = {}
    
    # Loop over each local variable in the calling frame
    for name, value in calling_frame.f_locals.items():
        # Create a dictionary to store the information for this variable
        var_info = {}
        
        # Get the size and location of the variable in memory
        var_info['size'] = sys.getsizeof(value)
        var_info['location'] = id(value)
        
        # If the variable is an np.ndarray, get its shape
        if isinstance(value, np.ndarray):
            var_info['shape'] = value.shape
        else:
            # Otherwise, set the shape attribute to 'NaN'
            var_info['shape'] = 'NaN'
        
        # Get the length of the variable
        var_info['length'] = len(value)
        
        # Get the first 128 characters of the variable as a string
        var_info['first_128_chars'] = str(value)[:128]
        
        # Add this variable's information to the dictionary of local variables
        local_vars[name] = var_info
    
    # Return the dictionary of local variables as a JSON string
    return json.dumps(local_vars)
