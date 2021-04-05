import docker
import tarfile
from pathlib import Path, PurePath


class DockerHandler:
    def __init__(self, image: str):
        self.image = image

        self.client = docker.from_env()
        self.container = None

    def run_container(self):
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
        folder = Path(host_path)
        folder.mkdir(exist_ok=True, parents=True)

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
        # Create the ls command
        ls_command = "ls {} -t".format(container_path)

        # List the elements
        elements = self.container.exec_run(ls_command).output.decode("utf-8")

        # Split according to line breaks and remove the last entry which is
        # always empty
        return elements.split("\n")[:-1]

    def copy_folder_from_container(self, container_path, host_path,
                                   file_filter):
        # Create necessary folders
        Path(host_path).mkdir(exist_ok=True, parents=True)

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
                for file_to_copy in file_filter:
                    if member.name.endswith(file_to_copy):
                        member.path = PurePath(member.path).name
                        members.append(member)

            tar.extractall(host_path, members)

        # Remove the tar file
        Path(tar_file_path).unlink()

    def run_command(self, command):
        self.container.exec_run(command)
