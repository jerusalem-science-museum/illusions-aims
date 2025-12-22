#!/usr/bin/env python


class Ansi(object):
    """
    ANSI terminal interface:
    Adds ANSI style colors to text.
    Sends ANSI terminal commands.
    """
    _ESCAPE = "\033["
    _STYLE_FORMAT = _ESCAPE + "%sm%s" + _ESCAPE + "0m"

    BRIGHT = "1"
    FAINT = "2"  # Not widely supported
    ITALIC = "3"
    UNDERLINE = "4"
    SLOW_BLINK = "5"
    FAST_BLINK = "6"  # Not widely supported

    BLACK = "30"
    RED = "31"
    GREEN = "32"
    BRIGHT = "1"
    FAINT = "2"  # Not widely supported
    ITALIC = "3"
    UNDERLINE = "4"
    SLOW_BLINK = "5"
    FAST_BLINK = "6"  # Not widely supported

    BLACK = "30"
    RED = "31"
    GREEN = "32"
    YELLOW = "33"
    BLUE = "34"
    MAGENTA = "35"
    CYAN = "36"
    WHITE = "37"
    DEFAULT = "39"

    BACKGROUND_BLACK = "40"
    BACKGROUND_RED = "41"
    BACKGROUND_GREEN = "42"
    BACKGROUND_YELLOW = "43"
    BACKGROUND_BLUE = "44"
    BACKGROUND_MAGENTA = "45"
    BACKGROUND_CYAN = "46"
    BACKGROUND_WHITE = "47"
    BACKGROUND_DEFAULT = "49"

    @classmethod
    def _send_command(cls, cmd, *args):
        print("%s%s%s" % (cls._ESCAPE, ";".join(map(str, args)), cmd))

    @classmethod
    def style(cls, text, *codes):
        """
        Adds ANSI style to the given text.
        :param text: Text to wrap.
        :param codes: ANSI codes.
        :return: The ANSI formatted text.
        """
        return cls._STYLE_FORMAT % (";".join(codes), text)

    @classmethod
    def set_cursor_position(cls, x, y):
        """
        :param x: > 0.
        :param y: > 0.
        """
        cls._send_command("H", x, y)

    @classmethod
    def move_cursor_horizontal(cls, x):
        """
        :param x: > 0 = right, x < 0 = left.
        """
        cls._send_command("D" if x < 0 else "C", abs(x))

    @classmethod
    def move_cursor_vertical(cls, y):
        """
        :param y: > 0 = down, y < 0 = up.
        """
        cls._send_command("A" if y < 0 else "B", abs(y))

    @classmethod
    def save_cursor_position(cls):
        cls._send_command("s")

    @classmethod
    def restore_cursor_position(cls):
        cls._send_command("u")

    @classmethod
    def clear_screen(cls):
        cls._send_command("2J")

    @classmethod
    def clear_screen_eol(cls):
        cls._send_command("k")



