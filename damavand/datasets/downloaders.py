import requests
import os
import time
import json
from zipfile import ZipFile
from rarfile import RarFile


def read_addresses():
    """
    Reads the addresses of datasets from the json file and returns as a dict
    """
    with open("damavand/damavand/datasets/addresses.json", "r") as f:
        data = json.load(f)
    return data


class ZipDatasetDownloader:
    def __init__(self, url):
        """
        Initializes the ZipDatasetDownloader with the specified URL.

                Parameters
        ----------
        url : str
                        The URL of the dataset zip file.

        Attributes
        ----------
                url : str
                        The URL of the dataset zip file.
        download_file_name : str
                        The file path where the dataset will be downloaded.
        extraction_path : str
                        The directory path where the dataset will be extracted.
        """
        self.url = url

    def download(self, download_file_name):
        """
        Downloads the dataset from the source URL into the specified file.

                Parameters
                ----------
                download_file_name : str
                        The file path where the dataset will be downloaded.
        """
        self.download_file_name = download_file_name
        response = requests.get(self.url)
        with open(self.download_file_name, "wb") as file:
            file.write(response.content)

    def extract(self, extraction_path):
        """
        Extracts the dataset from the downloaded zip file to the specified path.

                Parameters
                ----------
                extraction_path : str
            The directory path where the dataset will be extracted.
        """
        self.extraction_path = extraction_path
        with ZipFile(self.download_file_name, "r") as zObject:
            zObject.extractall(self.extraction_path)

    def download_extract(self, download_file_name, extraction_path):
        """
        Downloads the dataset to a specified file and extracts it to a specified path.

                Parameters
                ----------
                download_file_name : str
                        The file path where the dataset will be downloaded.
                extraction_path : str
                        The directory path where the dataset will be extracted.
        """
        self.download(download_file_name)
        self.extract(extraction_path)


class CwruDownloader:
    def __init__(self, files):
        """
        Initializes the CwruDownloader with the specified files.

        Parameters
        ----------
        files : dict
            A dictionary where keys are file names and values are the download links.

        Attributes
        ----------
        files : dict
            A dictionary where keys are file names and values are the download links.
        undownloaded : dict
            A dictionary whose keys are file names and values are the corresponding errors. This is used to keep track of the files that are not downloaded properly.
        """
        self.files = files
        self.undownloaded = {}

    def download(self, download_path, chunk_size=512, delay=1):
        """
        Downloads the dataset from the source URLs into the specified directory.

        Parameters
        ----------
        download_path : str
            The directory path where the dataset will be downloaded.
        chunk_size : int, optional
            The size of the chunks to write to the file, in bytes. The default is 512.
        delay : float, optional
            The delay between the requests to the server, in seconds. The default is 1.

        Notes
        -----
        This method does not raise any exceptions. Instead, it logs the errors and stores them in the undownloaded attribute.
        """
        self.download_path = download_path
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        for key in self.files:
            print(f"Downloading: {key}")
            try:
                response = requests.get(self.files[key], stream=True)
                if response.status_code == 200:
                    temp_file_path = os.path.join(
                        self.download_path, key.split(".")[0] + ".tmp"
                    )

                    with open(temp_file_path, mode="wb") as file:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                file.write(chunk)

                    os.rename(temp_file_path, os.path.join(self.download_path, key))
                    print(f"Downloaded: {key}")
                else:
                    print(
                        f"Error downloading {key}: Response status {response.status_code}"
                    )
                    self.undownloaded[key] = self.files[key]
                    os.remove(temp_file_path)

            except Exception as e:
                print(f"Error downloading {key}: {e}")
                self.undownloaded[key] = self.files[key]
                os.remove(temp_file_path)

            time.sleep(delay)

    def redownload(self, chunk_size=512, delay=1):
        """
        Redownloads the files that were not downloaded properly.

        Parameters
        ----------
        chunk_size : int, optional
            The size of the chunks to write to the file, in bytes. The default is 512.
        delay : float, optional
            The delay between the requests to the server, in seconds. The default is 1.

        Notes
        -----
        This method does not raise any exceptions. Instead, it logs the errors and stores them in the undownloaded attribute.
        """
        for key in list(self.undownloaded.keys()):
            print(f"Downloading: {key}")
            try:
                response = requests.get(self.files[key], stream=True)
                if response.status_code == 200:
                    temp_file_path = os.path.join(
                        self.download_path, key.split(".")[0] + ".tmp"
                    )

                    with open(temp_file_path, mode="wb") as file:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                file.write(chunk)

                    os.rename(temp_file_path, os.path.join(self.download_path, key))
                    self.undownloaded.pop(key)
                    print(f"Downloaded: {key}")

                else:
                    print(
                        f"Error downloading {key}: Response status {response.status_code}"
                    )
                    os.remove(temp_file_path)

            except Exception as e:
                print(f"Error downloading {key}: {e}")
                os.remove(temp_file_path)

            time.sleep(delay)


class PuDownloader:
    def __init__(self, files):
        """
        Initializes the downloader with the specified files.

        Parameters
        ----------
        files : dict
                A dictionary where keys are file names and values are the download links.
        """
        self.files = files

    def download(self, download_path, timeout=10):
        """
        Downloads the dataset files from the specified URLs into the given directory.

        Parameters
        ----------
        download_path : str
                The directory where the downloaded files will be saved.
        timeout : int, optional
                The maximum number of seconds to wait for a download to complete. The default is 10 seconds.

        Notes
        -----
        If the specified directory does not exist, it will be created. This method prints the file names as they are downloaded.
        """
        self.download_path = download_path
        if not os.path.exists(self.download_path):
            os.mkdir(self.download_path)
        for key in self.files.keys():
            for subkey in self.files[key].keys():
                print("Downloading: ", subkey)
                r = requests.get(self.files[key][subkey], stream=True, timeout=timeout)
                with open(self.download_path + subkey, "wb") as f:
                    f.write(r.content)

    def extract(self, extraction_path):
        """
        Extracts the dataset files from the downloaded rar files to the specified path.

        Parameters
        ----------
        extraction_path : str
            The directory path where the dataset will be extracted.
        """
        self.extraction_path = extraction_path
        for key in self.files.keys():
            for subkey in self.files[key]:
                print("Extracting: ", subkey)
                with RarFile(self.download_path + subkey) as file:
                    file.extractall(self.extraction_path)

    def download_extract(self, download_path, extraction_path, timeout=10):
        """
        Downloads the dataset files from the specified URLs into the given directory and extracts them into another directory.

        Parameters
        ----------
        download_path : str
            The directory where the downloaded files will be saved.
        extraction_path : str
            The directory path where the dataset will be extracted.
        timeout : int, optional
            The maximum number of seconds to wait for a download to complete. The default is 10 seconds.

        Notes
        -----
        If the specified directories do not exist, they will be created. This method prints the file names as they are downloaded and extracted.
        """
        self.download(download_path, timeout)
        self.extract(extraction_path)


class MaFaulDaDownloader:
    def __init__(self, files):
        """
        Initializes the MaFaulDaDownloader with the specified files.

        Parameters
        ----------
        files : dict of str
            A dictionary whose keys are file names and corresponding values are the download links.
        """
        self.files = files

    def download(self, download_path):
        """
        Downloads the dataset files from the specified URLs into the given directory.

        Parameters
        ----------
        download_path : str
            The directory where the downloaded files will be saved.

        Notes
        -----
        If the specified directory does not exist, it will be created. This method prints the file names as they are downloaded.
        """
        self.download_path = download_path
        if not os.path.exists(self.download_path):
            os.mkdir(self.download_path)
        for key in self.files:
            print("Downloading: ", key)
            r = requests.get(self.files[key], stream=True)
            with open(self.download_path + key, "wb") as f:
                f.write(r.content)

    def extract(self, extraction_path):
        """
        Extracts the dataset files from the downloaded zip files into the specified directory.

        Parameters
        ----------
        extraction_path : str
            The directory where the extracted files will be saved.

        Notes
        -----
        If the specified directory does not exist, it will be created. This method prints the file names as they are extracted.
        """
        self.extraction_path = extraction_path
        if not os.path.exists(self.download_path):
            os.mkdir(self.download_path)

        zip_files = [
            file for file in os.listdir(self.download_path) if file.endswith(".zip")
        ]
        for zip_file in zip_files:
            print("Extracting: ", zip_file)
            with ZipFile(self.download_path + zip_file, "r") as zObject:
                zObject.extractall(self.extraction_path)

    def download_extract(self, download_path, extraction_path):
        """
        Downloads the dataset files from the specified URLs into the given directory and extracts them into another directory.

        Parameters
        ----------
        download_path : str
            The directory where the downloaded files will be saved.
        extraction_path : str
            The directory path where the dataset will be extracted.

        Notes
        -----
        If the specified directories do not exist, they will be created. This method prints the file names as they are downloaded and extracted.
        """
        self.download(download_path)
        self.extract(extraction_path)


class Xjtu_suDownloader:
    def __init__(self, addresses):
        self.file_addresses = addresses

    def download(self, download_path, timeout=15):
        self.download_path = download_path
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

        for key, url in self.file_addresses.items():
            print(f"Downloading: {key}...")

            r = requests.get(url, stream=True, timeout=timeout)

            filename = f"archive.part{key[-2:]}.rar"
            with open(os.path.join(self.download_path, filename), "wb") as f:
                for chunk in r.iter_content(chunk_size=32768):
                    f.write(chunk)

        print("All parts downloaded. Checking sizes...")
        for f in os.listdir(self.download_path):
            print(f"{f}: {os.path.getsize(os.path.join(self.download_path, f))} bytes")

    def extract(self, extraction_path):
        files = sorted(
            [
                os.path.join(self.download_path, f)
                for f in os.listdir(self.download_path)
                if f.endswith(".rar")
            ]
        )
        with RarFile(files[0]) as rf:
            rf.extractall(path=extraction_path)
        print("Done!")

    def download_extract(self, download_path, extraction_path, timeout=15):
        self.download(download_path, timeout)
        self.extract(extraction_path)
