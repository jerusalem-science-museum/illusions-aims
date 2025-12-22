import logging
from ansi import Ansi


class Logor(object):
    """
    Colorize root log records.
    """

    def __init__(self, formats, level, color_map):
        logging.root.setLevel(level)
        handler = logging.StreamHandler()
        handler.setFormatter(ColorFormatter(formats[0], formats[1], color_map))
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)
        logging.root.addHandler(handler)


class ColorFormatter(logging.Formatter):
    """
    Colorize log records fields with the given color_map.
    """
    Colorize log records fields with the given color_map.
    """

    def __init__(self, fmt=None, datefmt=None, color_map=None):
        if color_map is None:
            color_map = {}
        self._fields_codes = {k: v for k, v in color_map.items() if type(k) is str}
        self._levels_codes = {k: v for k, v in color_map.items() if type(k) is int}
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        record.msg = Ansi.style(record.msg, *self._levels_codes.get(record.levelno, (Ansi.WHITE, Ansi.BRIGHT)))
        for field, codes in self._fields_codes.items():
            setattr(record, field, Ansi.style(getattr(record, field), *codes))
        return logging.Formatter.format(self, record)
