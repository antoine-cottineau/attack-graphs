import random
from pathlib import Path


def get_file_extension(file) -> str:
    """Returns the extension of the file.

    Args:
        file (str|pathlib.PosixPath): A string or pathlib.PosixPath object
    leading to the location of the file.

    Returns:
        str: The extension of the file.
    """
    pathlib_file = Path(file)
    return pathlib_file.suffix[1:]


def create_parent_folders(file):
    """Creates the parent folders of the file.

    Args:
        file (str|pathlib.PosixPath): A string or pathlib.PosixPath object
    leading to the location of the file.
    """
    pathlib_file = Path(file).parent
    pathlib_file.mkdir(exist_ok=True, parents=True)


def create_random_color() -> str:
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return "rgb({},{},{})".format(r, g, b)
