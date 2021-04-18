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
