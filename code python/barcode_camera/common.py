import os
import sys
import os.path
import logging


# add BASIC_PATH (GitHub clone path) to PYTHONPATH
BASIC_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
sys.path.append(BASIC_PATH)


from ansi import Ansi

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

IS_WINDOWS = (os.name == 'nt')

LOGOR_LEVEL = logging.INFO
LOGOR_FORMATS = ('%(asctime)s %(name)s %(levelname)s: %(message)s', DATETIME_FORMAT)
LOGOR_COLOR_MAP = {
    logging.CRITICAL: (Ansi.YELLOW, Ansi.BACKGROUND_RED,),
    logging.ERROR: (Ansi.RED, Ansi.FAINT),}

DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

IS_WINDOWS = (os.name == 'nt')

LOGOR_LEVEL = logging.INFO
LOGOR_FORMATS = ('%(asctime)s %(name)s %(levelname)s: %(message)s', DATETIME_FORMAT)
LOGOR_COLOR_MAP = {
    logging.CRITICAL: (Ansi.YELLOW, Ansi.BACKGROUND_RED,),
    logging.ERROR: (Ansi.RED, Ansi.FAINT),
    logging.WARNING: (Ansi.YELLOW, Ansi.FAINT),
    logging.INFO: (Ansi.CYAN, Ansi.FAINT),
    logging.DEBUG: (Ansi.CYAN, Ansi.BRIGHT),
    'name': (Ansi.GREEN, Ansi.FAINT),
    'levelname': (Ansi.MAGENTA, Ansi.FAINT),
}
