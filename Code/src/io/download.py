# SUSY: https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz
# DOTA2: https://archive.ics.uci.edu/ml/machine-learning-databases/00367/dota2Dataset.zip
# covertype: https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype

# 

# .data.gz
from queue import  Queue
import os
import io
from urllib import request
from src.conf.settings import CONFIG

logger = CONFIG.get_logger()


class Download:

    def __init__(self, url=None):
        self.dl_queue = Queue()
        self.dest = ""
        self.status = 0
        self.next = None

        if url is None:
            for key, value in CONFIG.URLS.items():
                self.dl_queue.put((key, value))
        else:
            self.dl_queue.put((' ', url))

    def progress(self):
        pass

    def add_link(self):
        pass

    def start(self):
        while not self.dl_queue.empty():
            self.next = self.dl_queue.get()
            self._download()

    def set_destination(self, path, use_root=True):
        if use_root:
            self.dest = os.path.join(CONFIG.ROOT_DIR, path)
        else:
            self.dest = path

    def status(self) -> bool:
        return self.dl_queue.empty()

    def _download(self):
        name, url = self.next
        if not os.path.exists(os.path.join(CONFIG.ROOT_DIR, "data", name)):
            print("Downloading " + name)
            data = request.urlopen(url)
            length = data.getheader('content-length')
            fname = url.split("/")[-1]
            if length:
                length = int(length)
                blocksize = max(4096, length//100)
                print(str(length))

                writeable = io.BytesIO()
                size = 0
                while True:
                    buf1 = data.read(blocksize)
                    if not buf1:
                        break
                    writeable.write(buf1)
                    size += len(buf1)
                    if length:
                        logger.info('{:.2f}%\r  done'.format(size/length))
            else:
                writeable = data.read()

            os.makedirs(os.path.join(CONFIG.ROOT_DIR, "data", name))
            with open(os.path.join(CONFIG.ROOT_DIR, "data", name, fname), 'wb') as file:
                if isinstance(writeable, io.BytesIO):
                    print("Writing Bytes")
                    file.write(writeable.getvalue())
                else:
                    print("Writing File")
                    file.write(writeable)
        else:
            print("Directory for that Data Set already exists. Please check the containing files and remove the directory to proceed: \n" + os.path.join(CONFIG.ROOT_DIR, "data" ,name))
