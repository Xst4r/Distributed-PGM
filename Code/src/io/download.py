# SUSY: https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz
# DOTA2: https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip
# covertype: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype

# 

# .data.gz
from queue import  Queue
import os
from urllib import request
from src.conf.modes import ROOT_DIR, URLS

class Download:

    def __init__(self):
        self.dl_queue = Queue()
        self.dest = ""
        self.status = 0
        self.next = None

        for key, value in URLS.items():
            self.dl_queue.put((key, value))

    def progress(self):
        pass

    def add_link(self):
        pass

    def start(self):
        while self.dl_queue.not_empty:
            self.next = self.dl_queue.get()
            self._download()

    def set_destination(self, path, use_root=True):
        if use_root:
            self.dest = os.path.join(ROOT_DIR, path)
        else:
            self.dest = path

    def status(self) -> bool:
        return self.dl_queue.empty()

    def _download(self):
        name, url = self.next
        print("Dowloading " + name)
        data = request.urlopen(url)
        writeable = data.read()
        fname = url.split("/")[-1]

        if not os.path.exists(os.path.join(ROOT_DIR, "data" ,name)):
            os.makedirs(os.path.join(ROOT_DIR, "data" ,name))
            with open(os.path.join(ROOT_DIR, "data" ,name, fname), 'wb') as file:
                file.write(writeable)
        else:
            print("Directory for that Data Set already exists. Please check the containing files and remove the directory to proceed: \n" + os.path.join(ROOT_DIR, "data" ,name))
