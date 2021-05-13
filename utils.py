import random
from pathlib import Path
from typing import List


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


def create_folders(path):
    pathlib_file = Path(path)
    pathlib_file.mkdir(exist_ok=True, parents=True)


def create_parent_folders(file):
    """Creates the parent folders of the file.

    Args:
        file (str|pathlib.PosixPath): A string or pathlib.PosixPath object
    leading to the location of the file.
    """
    pathlib_file = Path(file).parent
    create_folders(pathlib_file)


def list_files_in_directory(directory) -> List[Path]:
    files = Path(directory).glob("**/*")
    return [file for file in files if file.is_file()]


def create_random_color() -> str:
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return "rgb({},{},{})".format(r, g, b)


def sanitize(text: str) -> str:
    return text.replace(" ", "_").lower()
