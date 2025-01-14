import logging

def initialize_logger():
    """
    Initializes and configures a logger for the application.
    This function sets up a logger named 'xml_fixer' with a DEBUG logging level.
    It creates a console handler that outputs log messages to the console, also
    set to the DEBUG level. The log messages are formatted to include the timestamp,
    logger name, log level, and the message.
    Returns:
        logging.Logger: Configured logger instance.
    Example:
        logger = Initialize_logger()
    """
    # Initialize logger
    logger = logging.getLogger('custom-batch-inference')
    logger.setLevel(logging.DEBUG)

    # Create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to console handler
    ch.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(ch)

    # Example usage
    logger.debug('Logger initialized successfully.')
    return logger