# config.py

import os  # Importing the os module to interact with the operating system's file system

# BASE_DIR is set to the directory where this config.py file is located.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The paths dictionary define a key that represents the absolute path to that directory, based on BASE_DIR.
paths = {
    'data': os.path.join(BASE_DIR, 'data'),                     # Path to the 'data' directory, where raw (labels and images) should be stored. This need to be created and data need to be stored.
    'models': os.path.join(BASE_DIR, 'models'),                  # Path to the 'models' directory, where machine learning models are saved.
    'results': os.path.join(BASE_DIR, 'results'),                  # Path to the 'results' directory, where output results are stored. This will be created automatically.
    'preprocessing': os.path.join(BASE_DIR, 'preprocessing'),         # Path to the 'preprocessing' directory, for storing preprocessing results and image data. This will be created automatically.
    'log_file': os.path.join(BASE_DIR, 'logs', 'process.log')          # Path to the 'logs' directory, where log files will be saved. This will also be created automatically.
}

# Ensure the logs directory exists, creating it if necessary
os.makedirs(os.path.dirname(paths['log_file']), exist_ok=True)