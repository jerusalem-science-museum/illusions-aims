import logging
import utils
from logor import Logor


class App(object):

    _app_logger = logging.getLogger('app')

    def __init__(self, constants=None, spam_loggers=None):
        self._modules = []
        if constants is not None:
            self._modules.append(constants)
        if spam_loggers is not None:
            for i in spam_loggers:
                try:
                    logging.getLogger(i).setLevel(logging.WARNING)
                except:
                    pass
        if utils.hasattrs(constants, 'LOGOR_FORMATS', 'LOGOR_LEVEL', 'LOGOR_COLOR_MAP'):
            Logor(constants.LOGOR_FORMATS, constants.LOGOR_LEVEL, constants.LOGOR_COLOR_MAP)
        self._app_logger.log(logging.root.level, 'App started, Logging level: %s', logging.getLevelName(logging.root.level))

    def __exit__(self):
        self._app_logger.warn('App ended\n')


