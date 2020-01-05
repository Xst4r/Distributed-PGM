from src.util.conf import ROOT_DIR

import os, zipfile, gzip, shutil, tarfile
import logging

class Extract:

    def __init__(self):
        self.path = os.path.join(ROOT_DIR, "data")

    def extract_all(self):
        os.chdir(self.path)
        for dir in os.listdir(self.path):
            dirs = os.listdir(dir)
            for file in dirs:
                if file.endswith(".gz"):
                    with gzip.open(os.path.join(dir, file), 'rb') as gzipper:
                        name = file.split(".")[0]
                        ending = file.split(".")[-2]
                        with open(os.path.join(self.path, dir, name + "." + ending), 'wb') as gzip_out:
                            shutil.copyfileobj(gzipper, gzip_out)
                            print("Done")
                if file.endswith(".zip"):
                    try:
                        with zipfile.ZipFile(os.path.join(dir, file), 'r') as zipObj:
                            zipObj.extractall(path=dir)
                    except FileExistsError:
                        logging.info("File has already been extracted")
