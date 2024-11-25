import logging

def setup_logging(level = logging.DEBUG):
    """Setup logging."""
    logger = logging.getLogger()

    formatter = logging.Formatter(
        '[%(asctime)s] [%(name)s]: %(levelname)s %(message)s'
    )

    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

