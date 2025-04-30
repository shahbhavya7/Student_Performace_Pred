import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" # this will create a log file with the current date and time in the format of month_day_year_hour_minute_second.log
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE) # this will create a path for the log file in the logs folder in the current working directory
os.makedirs(logs_path,exist_ok=True) # this will create the logs folder if it does not exist, exist_ok=True means it will not raise an error if the folder already exists

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)  # this will create the full path for the log file

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,


) # this will set the basic configuration for the logging module, it will create a log file with the name LOG_FILE_PATH, the format of the log file will be as 
# specified in the format parameter, and the level of logging will be INFO, which means it will log all messages with level INFO and above (WARNING, ERROR, CRITICAL)

