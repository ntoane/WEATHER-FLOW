import logging
""" 
logger class that uses Python's logging module as a flexible logging framework.
It allows you to log messages at different levels, define custom log handlers, and configure log formatting. 
"""
class modelLogger:
    def __init__(self, model_name, log_file):
        """
        Initializes the ModelLogger class.

        Args:
            model_name (str): Name of the model.
            log_file (str): Path to the log file.
        """
        self.logger = logging.getLogger(model_name)
        self.logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
    
    def info(self, message):
        """
        Logs an info-level message.

        Args:
            message (str): Log message.
        """
        self.logger.info(message)
    
    def debug(self, message):
        """
        Logs a debug-level message.

        Args:
            message (str): Log message.
        """
        self.logger.debug(message)
    
    def warning(self, message):
        """
        Logs a warning-level message.

        Args:
            message (str): Log message.
        """
        self.logger.warning(message)
    
    def error(self, message):
        """
        Logs an error-level message.

        Args:
            message (str): Log message.
        """
        self.logger.error(message)
    
    def close(self):
        """
        Closes the log file handlers.
        """
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)




# # Create a logger specific to your GWN model
# gwn_logger = logging.getLogger('gwn')
# gwn_logger.setLevel(logging.DEBUG)

# # Create a file handler to log messages to a file
# file_handler = logging.FileHandler('gwn_logs.txt')
# file_handler.setLevel(logging.DEBUG)

# # Create a formatter to define the log message format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # Set the formatter for the file handler
# file_handler.setFormatter(formatter)

# # Add the file handler to the logger
# gwn_logger.addHandler(file_handler)

# # Log some messages with different log levels
# gwn_logger.info('GWN model initialized.')
# gwn_logger.debug('Debug message: This is a debug log.')
# gwn_logger.warning('Warning message: The input data may be incomplete.')

# # Create a logger specific to your TCN model
# tcn_logger = logging.getLogger('tcn')
# tcn_logger.setLevel(logging.INFO)

# # Create another file handler for the TCN model
# tcn_file_handler = logging.FileHandler('tcn_logs.txt')
# tcn_file_handler.setLevel(logging.INFO)
# tcn_file_handler.setFormatter(formatter)
# tcn_logger.addHandler(tcn_file_handler)

# # Log some messages with different log levels
# tcn_logger.info('TCN model training started.')
# tcn_logger.debug('Debug message: This is a debug log.')
# tcn_logger.error('Error message: An error occurred during training.')

# # Close the log file handlers
# file_handler.close()
# tcn_file_handler.close()