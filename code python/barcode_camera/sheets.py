import os
import os.path
import logging
import pygsheets
from common import IS_WINDOWS


class Sheets(object):
    """
    Wrapper for google sheets using pygsheets.
    """
    SPAM_LOGGERS = ('googleapiclient.discovery', 'oauth2client.transport', 'oauth2client.crypt', 'oauth2client.client')
    _logger = logging.getLogger('google.sheets')


    def __init__(self, service_account):
        self._sheets = pygsheets.authorize(service_file=service_account, no_cache=IS_WINDOWS)
        self._worksheet_cache = {}

    def append_worksheet_table(self, sheet_name, worksheet, *values):
        _key = (sheet_name, worksheet)
        if _key not in self._worksheet_cache:
            sheet = self._sheets.open(sheet_name)
            worksheet = sheet.worksheet_by_title(worksheet)
            if worksheet is None:
                raise KeyError('no %s worksheet in %s sheet' % _key[::-1])
            self._logger.debug('open %s.%s worksheet', *_key)
            self._worksheet_cache[_key] = worksheet
        self._worksheet_cache[_key].append_table(values=values)
