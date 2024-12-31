_color_map = {
    "blue": "\033[94m",
    "green": "\033[92m",
    "red": "\033[91m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "magenta": "\033[95m",
    "white": "\033[97m",
    "orange": "\033[38;5;208m",
    "pink": "\033[38;5;207m",
    "purple": "\033[38;5;99m",
}


def _print_color(
        color: str,
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    values = list(values)
    values[0] = _color_map[color] + str(values[0])
    values[-1] = str(values[-1]) + "\033[0m"
    print(*values, sep=sep, end=end, file=file, flush=flush)


def print_blue(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("blue", *values, sep=sep, end=end, file=file, flush=flush)


def print_green(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("green", *values, sep=sep, end=end, file=file, flush=flush)


def print_red(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("red", *values, sep=sep, end=end, file=file, flush=flush)


def print_yellow(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("yellow", *values, sep=sep, end=end, file=file, flush=flush)


def print_cyan(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("cyan", *values, sep=sep, end=end, file=file, flush=flush)


def print_magenta(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("magenta", *values, sep=sep, end=end, file=file, flush=flush)


def print_white(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("white", *values, sep=sep, end=end, file=file, flush=flush)


def print_orange(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("orange", *values, sep=sep, end=end, file=file, flush=flush)


def print_pink(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("pink", *values, sep=sep, end=end, file=file, flush=flush)


def print_purple(
        *values: object,
        sep=" ",
        end="\n",
        file=None,
        flush=False
):
    _print_color("purple", *values, sep=sep, end=end, file=file, flush=flush)


if __name__ == '__main__':
    print_blue("Hello, World!", 123)
    print_cyan("Hello, World!", 123)
    print_magenta("Hello, World!", 123)
    print_yellow("Hello, World!", 123)
    print_red("Hello, World!", 123)
    print_orange("Hello, World!", 123)
    print_pink("Hello, World!", 123)
    print_purple("Hello, World!", 123)
