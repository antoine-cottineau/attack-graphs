import docker
import tarfile
from pathlib import Path, PurePath


class DockerHandler:
    """
    Class used to run some code in a Docker container. It can be useful when
    the code needs a particular environment to be run.

    :param str image: The docker image of the container.
    """
    def __init__(self, image: str):
        self.image = image

        self.client = docker.from_env()
        self.container = None

    def run_container(self):
        """
        Run a container with the image specified during class instanciation. If
        a container with this image is already running, it is used again.
        """
        # A container is already stored in the variable self.container
        if self.container:
            return

        # We look for running containers with image self.image
        for container in self.client.containers.list():
            if container.attrs["Config"]["Image"] == self.image:
                self.container = container
                return

        # Otherwise, we run a new container
        if not self.container:
            self.container = self.client.containers.run(self.image,
                                                        detach=True)

    def transfer_folder(self, host_path, container_path, tar_file_name):
        """
        Transfer a folder from the host to the container.

        :param str host_path: The path to the folder to transfer.
        :param str container_path: The path indicating where to copy the folder
        in the container.
        :param str tar_file_name: The name of the tar archive to transfer.
        """
        folder = Path(host_path)
        folder.mkdir(parents=True, exist_ok=True)

        # Create a temporary archive with the files in folder
        tar_file_path = "temp/{}.tar".format(tar_file_name)
        with tarfile.open(tar_file_path, "w") as tar:
            for f in folder.iterdir():
                if f.is_file():
                    tar.add(f)

        # Transfer the archive to the container
        data = open(tar_file_path, "rb").read()
        self.container.put_archive(container_path, data)

        # Delete the temporary archive
        Path(tar_file_path).unlink()

    def list_elements_in_container(self, container_path):
        """
        List all elements in a folder in the container.

        :param str container_path: The path of the folder to explore in the
        container.
        :return list elements: A list of the elements in the folder, sorted
        from most recent to oldest.
        """
        # Create the ls command
        ls_command = "ls {} -t".format(container_path)

        # List the elements
        elements = self.container.exec_run(ls_command).output.decode("utf-8")

        # Split according to line breaks and remove the last entry which is
        # always empty
        return elements.split("\n")[:-1]

    def copy_folder_from_container(self, container_path, host_path,
                                   file_filter):
        """
        Copy a folder from the container to the host.

        :param str container_path: The path of the folder to copy.
        :param str host_path: The path indicating where to copy the folder in
        the host.
        :param list file_filter: A list of the files that need to be copied
        from the folder.
        """
        # Create necessary folders
        Path(host_path).mkdir(parents=True, exist_ok=True)

        # Copy the folder to a tar file
        stream, stats = self.container.get_archive(container_path)

        tar_file_path = "temp/{}.tar".format(stats["name"])
        with open(tar_file_path, "wb") as f:
            for chunk in stream:
                f.write(chunk)

        # Extract the content of the tar file
        files_to_copy = []
        for f in file_filter:
            files_to_copy.append("{}/{}".format(container_path, f))

        with tarfile.open(tar_file_path) as tar:
            members = []

            # Only keep the basename of the files
            for member in tar.getmembers():
                if member.name in files_to_copy:
                    member.path = PurePath(member.path).name
                    members.append(member)

            tar.extractall(host_path, members)

        # Remove the tar file
        Path(tar_file_path).unlink()

    def run_command(self, command):
        """
        Run a command in the container
        :param str command: The command to run.
        """
        self.container.exec_run(command)
