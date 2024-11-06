"""
Main script for the program.
Imports config variables, initializes loggers, and tests the matmul function.
"""

from utils import logger
from utils import matmul


def main():
    print('\nStarting program...')

    logger.config_logger()

    matmul.matmul([2, 2], [3, 3])


if __name__ == '__main__':
    main()
